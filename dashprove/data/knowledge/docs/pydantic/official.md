# Pydantic - Data Validation Using Python Type Hints

Pydantic is the most widely used data validation library for Python. It uses Python type hints to validate data and provides powerful features for parsing, serialization, and JSON schema generation.

## Installation

```bash
# Basic installation
pip install pydantic

# With email validation
pip install "pydantic[email]"

# Using uv
uv add pydantic
```

## Basic Usage

### Defining Models

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class User(BaseModel):
    id: int
    name: str
    email: str
    signup_ts: Optional[datetime] = None
    friends: list[int] = []

# Create instance
user = User(
    id=123,
    name="John Doe",
    email="john@example.com",
    signup_ts="2024-01-15 12:00:00",
    friends=[1, 2, 3]
)

print(user.id)  # 123
print(user.model_dump())  # Dict representation
print(user.model_dump_json())  # JSON string
```

### Data Validation

```python
from pydantic import BaseModel, ValidationError

class Item(BaseModel):
    name: str
    price: float
    quantity: int

try:
    item = Item(name="Widget", price="invalid", quantity=10)
except ValidationError as e:
    print(e.errors())
    # [{'type': 'float_parsing', 'loc': ('price',), ...}]
```

## Field Types and Constraints

### Built-in Constraints

```python
from pydantic import BaseModel, Field
from typing import Annotated

class Product(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    price: float = Field(gt=0, description="Price must be positive")
    quantity: int = Field(ge=0, le=1000)
    sku: str = Field(pattern=r'^[A-Z]{3}-\d{4}$')

    # Alternative with Annotated
    discount: Annotated[float, Field(ge=0, le=1)] = 0.0
```

### String Constraints

```python
from pydantic import BaseModel, Field

class Text(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    slug: str = Field(pattern=r'^[a-z0-9-]+$')
    content: str = Field(default="")
```

### Numeric Constraints

```python
from pydantic import BaseModel, Field

class Numbers(BaseModel):
    positive: int = Field(gt=0)      # greater than
    non_negative: int = Field(ge=0)  # greater than or equal
    negative: int = Field(lt=0)      # less than
    limited: int = Field(le=100)     # less than or equal
    multiple_of: int = Field(multiple_of=5)
```

## Custom Validators

### Field Validators

```python
from pydantic import BaseModel, field_validator

class User(BaseModel):
    name: str
    email: str

    @field_validator('name')
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.title()

    @field_validator('email')
    @classmethod
    def email_must_be_valid(cls, v: str) -> str:
        if '@' not in v:
            raise ValueError('Invalid email')
        return v.lower()
```

### Model Validators

```python
from pydantic import BaseModel, model_validator

class DateRange(BaseModel):
    start_date: datetime
    end_date: datetime

    @model_validator(mode='after')
    def check_dates(self) -> 'DateRange':
        if self.start_date >= self.end_date:
            raise ValueError('start_date must be before end_date')
        return self
```

### Before/After Validation

```python
from pydantic import BaseModel, field_validator

class Data(BaseModel):
    value: str

    @field_validator('value', mode='before')
    @classmethod
    def convert_to_string(cls, v):
        return str(v) if v is not None else ''

    @field_validator('value', mode='after')
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()
```

## Nested Models

```python
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str
    country: str

class Company(BaseModel):
    name: str
    address: Address

company = Company(
    name="Acme Inc",
    address={"street": "123 Main St", "city": "NYC", "country": "USA"}
)
```

## Model Configuration

```python
from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        str_min_length=1,
        validate_assignment=True,
        extra='forbid',  # 'allow', 'ignore', or 'forbid'
        frozen=False,
        populate_by_name=True,
    )

    name: str
    email: str
```

## JSON Schema Generation

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float

# Generate JSON Schema
schema = Item.model_json_schema()
print(schema)
# {
#   'properties': {
#     'name': {'title': 'Name', 'type': 'string'},
#     'price': {'title': 'Price', 'type': 'number'}
#   },
#   'required': ['name', 'price'],
#   'title': 'Item',
#   'type': 'object'
# }
```

## Serialization

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    password: str

user = User(id=1, name="John", password="secret")

# To dict
user.model_dump()
user.model_dump(exclude={'password'})
user.model_dump(include={'id', 'name'})

# To JSON
user.model_dump_json()
user.model_dump_json(indent=2)

# Exclude unset values
user.model_dump(exclude_unset=True)
```

## Type Coercion

```python
from pydantic import BaseModel

class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True)
    value: int

# Strict mode - no coercion
StrictModel(value=1)    # OK
StrictModel(value="1")  # ValidationError

class LooseModel(BaseModel):
    value: int

# Default - with coercion
LooseModel(value="1")  # OK, coerces to int 1
```

## Special Types

```python
from pydantic import BaseModel, EmailStr, HttpUrl, SecretStr
from datetime import datetime
from typing import Literal

class Config(BaseModel):
    email: EmailStr
    website: HttpUrl
    api_key: SecretStr
    environment: Literal['dev', 'staging', 'prod']
    created_at: datetime
```

## Computed Fields

```python
from pydantic import BaseModel, computed_field

class Rectangle(BaseModel):
    width: float
    height: float

    @computed_field
    @property
    def area(self) -> float:
        return self.width * self.height
```

## Generic Models

```python
from pydantic import BaseModel
from typing import Generic, TypeVar

T = TypeVar('T')

class Response(BaseModel, Generic[T]):
    data: T
    status: str

# Use with specific type
class UserResponse(Response[User]):
    pass
```

## Integration with FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.post("/items/")
async def create_item(item: Item):
    return item
```

## Best Practices

1. **Use type hints** - They're required for validation
2. **Define constraints with Field()** - Clear and explicit
3. **Use validators sparingly** - Prefer built-in constraints
4. **Set extra='forbid'** - Catch typos in input
5. **Use model_config** - Consistent settings per model
6. **Keep models simple** - Split complex models

## Migration from V1 to V2

```python
# V1 (deprecated)
class User(BaseModel):
    class Config:
        extra = 'forbid'

# V2 (current)
class User(BaseModel):
    model_config = ConfigDict(extra='forbid')

# Method changes
user.dict()       # V1
user.model_dump() # V2

user.json()            # V1
user.model_dump_json() # V2
```

## Links

- Documentation: https://docs.pydantic.dev/
- GitHub: https://github.com/pydantic/pydantic
- PyPI: https://pypi.org/project/pydantic/
