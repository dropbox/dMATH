# Instructor Cookbooks[¶][1]

* **Text Processing**
  
  Extract structured information from text documents
  
  [ View Recipes][2]
* **Multi-Modal**
  
  Work with images and other media types
  
  [ View Recipes][3]
* **Data Tools**
  
  Integrate with databases and data processing tools
  
  [ View Recipes][4]
* **Deployment**
  
  Options for local and cloud deployment
  
  [ View Recipes][5]

Our cookbooks demonstrate how to use Instructor to solve real-world problems with structured
outputs. Each example includes complete code and explanations to help you implement similar
solutions in your own projects.

## Text Processing[¶][6]

### Classification Examples[¶][7]

──────────────────────────────────┬────────────────────────────────────────┬────────────────────────
Example                           │Description                             │Use Case                
──────────────────────────────────┼────────────────────────────────────────┼────────────────────────
[Single Classification][8]        │Basic classification with a single      │Content categorization  
                                  │category                                │                        
──────────────────────────────────┼────────────────────────────────────────┼────────────────────────
[Multiple Classification][9]      │Handling multiple classification        │Multi-label document    
                                  │categories                              │tagging                 
──────────────────────────────────┼────────────────────────────────────────┼────────────────────────
[Enum-Based Classification][10]   │Using Python enums for structured       │Standardized taxonomies 
                                  │classification                          │                        
──────────────────────────────────┼────────────────────────────────────────┼────────────────────────
[Batch Classification][11]        │Process multiple items efficiently      │High-volume text        
                                  │                                        │processing              
──────────────────────────────────┼────────────────────────────────────────┼────────────────────────
[Batch Classification with        │Using LangSmith for batch processing    │Performance monitoring  
LangSmith][12]                    │                                        │                        
──────────────────────────────────┼────────────────────────────────────────┼────────────────────────
[Local Classification][13]        │Classification without external APIs    │Offline processing      
──────────────────────────────────┴────────────────────────────────────────┴────────────────────────

### Information Extraction[¶][14]

────────────────────────────┬───────────────────────────────────────┬────────────────────
Example                     │Description                            │Use Case            
────────────────────────────┼───────────────────────────────────────┼────────────────────
[Entity Resolution][15]     │Identify and disambiguate entities     │Name standardization
────────────────────────────┼───────────────────────────────────────┼────────────────────
[Contact Information][16]   │Extract structured contact details     │CRM data entry      
────────────────────────────┼───────────────────────────────────────┼────────────────────
[PII Sanitization][17]      │Detect and redact sensitive information│Privacy compliance  
────────────────────────────┼───────────────────────────────────────┼────────────────────
[Citation Extraction][18]   │Accurately extract formatted citations │Academic research   
────────────────────────────┼───────────────────────────────────────┼────────────────────
[Action Items][19]          │Extract tasks from text                │Meeting follow-ups  
────────────────────────────┼───────────────────────────────────────┼────────────────────
[Search Query               │Structure complex search queries       │Search enhancement  
Processing][20]             │                                       │                    
────────────────────────────┴───────────────────────────────────────┴────────────────────

### Document Processing[¶][21]

───────────────────────────────┬─────────────────────────────────────────┬──────────────────────────
Example                        │Description                              │Use Case                  
───────────────────────────────┼─────────────────────────────────────────┼──────────────────────────
[Document Segmentation][22]    │Divide documents into meaningful sections│Long-form content analysis
───────────────────────────────┼─────────────────────────────────────────┼──────────────────────────
[Planning and Tasks][23]       │Break down complex queries into subtasks │Project management        
───────────────────────────────┼─────────────────────────────────────────┼──────────────────────────
[Knowledge Graph               │Create relationship graphs from text     │Information visualization 
Generation][24]                │                                         │                          
───────────────────────────────┼─────────────────────────────────────────┼──────────────────────────
[Knowledge Graph Building][25] │Build and query knowledge graphs         │Semantic data modeling    
───────────────────────────────┼─────────────────────────────────────────┼──────────────────────────
[Chain of Density][26]         │Implement iterative summarization        │Content distillation      
───────────────────────────────┴─────────────────────────────────────────┴──────────────────────────

## Multi-Modal Examples[¶][27]

### Vision Processing[¶][28]

────────────────────────────────┬───────────────────────────────────────┬────────────────────────
Example                         │Description                            │Use Case                
────────────────────────────────┼───────────────────────────────────────┼────────────────────────
[Table Extraction][29]          │Convert image tables to structured data│Data entry automation   
────────────────────────────────┼───────────────────────────────────────┼────────────────────────
[Table Extraction with          │Advanced table extraction              │Complex table processing
GPT-4][30]                      │                                       │                        
────────────────────────────────┼───────────────────────────────────────┼────────────────────────
[Receipt Information][31]       │Extract data from receipt images       │Expense management      
────────────────────────────────┼───────────────────────────────────────┼────────────────────────
[Slide Content Extraction][32]  │Convert slides to structured text      │Presentation analysis   
────────────────────────────────┼───────────────────────────────────────┼────────────────────────
[Image to Ad Copy][33]          │Generate ad text from images           │Marketing automation    
────────────────────────────────┼───────────────────────────────────────┼────────────────────────
[YouTube Clip Analysis][34]     │Extract info from video clips          │Content moderation      
────────────────────────────────┴───────────────────────────────────────┴────────────────────────

### Multi-Modal Processing[¶][35]

───────────────────────┬────────────────────────────────────┬────────────────────
Example                │Description                         │Use Case            
───────────────────────┼────────────────────────────────────┼────────────────────
[Gemini                │Process text, images, and other data│Mixed-media analysis
Multi-Modal][36]       │                                    │                    
───────────────────────┴────────────────────────────────────┴────────────────────

## Data Tools[¶][37]

### Database Integration[¶][38]

─────────────────────────┬────────────────────────────────────────┬──────────────────
Example                  │Description                             │Use Case          
─────────────────────────┼────────────────────────────────────────┼──────────────────
[SQLModel                │Store AI-generated data in SQL databases│Persistent storage
Integration][39]         │                                        │                  
─────────────────────────┼────────────────────────────────────────┼──────────────────
[Pandas DataFrame][40]   │Work with structured data in Pandas     │Data analysis     
─────────────────────────┴────────────────────────────────────────┴──────────────────

### Streaming and Processing[¶][41]

─────────────────────────────────┬───────────────────────────────────┬────────────────────────
Example                          │Description                        │Use Case                
─────────────────────────────────┼───────────────────────────────────┼────────────────────────
[Partial Response Streaming][42] │Stream partial results in real-time│Interactive applications
─────────────────────────────────┼───────────────────────────────────┼────────────────────────
[Self-Critique and               │Implement self-assessment          │Quality improvement     
Correction][43]                  │                                   │                        
─────────────────────────────────┴───────────────────────────────────┴────────────────────────

### API Integration[¶][44]

─────────────────────────────────────┬──────────────────────────────────┬─────────────────────
Example                              │Description                       │Use Case             
─────────────────────────────────────┼──────────────────────────────────┼─────────────────────
[Content Moderation][45]             │Implement content filtering       │Trust & safety       
─────────────────────────────────────┼──────────────────────────────────┼─────────────────────
[Cost Optimization with Batch        │Reduce API costs                  │Production efficiency
API][46]                             │                                  │                     
─────────────────────────────────────┼──────────────────────────────────┼─────────────────────
[Few-Shot Learning][47]              │Use contextual examples in prompts│Performance tuning   
─────────────────────────────────────┴──────────────────────────────────┴─────────────────────

### Observability & Tracing[¶][48]

─────────────────────┬───────────────────────────┬─────────────────────────
Example              │Description                │Use Case                 
─────────────────────┼───────────────────────────┼─────────────────────────
[Langfuse            │Open-source LLM engineering│Observability & Debugging
Tracing][49]         │                           │                         
─────────────────────┴───────────────────────────┴─────────────────────────

## Deployment Options[¶][50]

### Model Providers[¶][51]

───────────────────────────┬─────────────────────────────┬─────────────────────────
Example                    │Description                  │Use Case                 
───────────────────────────┼─────────────────────────────┼─────────────────────────
[Groq Cloud API][52]       │High-performance inference   │Low-latency applications 
───────────────────────────┼─────────────────────────────┼─────────────────────────
[Mistral/Mixtral           │Open-source model integration│Cost-effective deployment
Models][53]                │                             │                         
───────────────────────────┼─────────────────────────────┼─────────────────────────
[IBM watsonx.ai][54]       │Enterprise AI platform       │Business applications    
───────────────────────────┴─────────────────────────────┴─────────────────────────

### Local Deployment[¶][55]

───────────────────────┬────────────────────────┬────────────────────────────
Example                │Description             │Use Case                    
───────────────────────┼────────────────────────┼────────────────────────────
[Ollama                │Local open-source models│Privacy-focused applications
Integration][56]       │                        │                            
───────────────────────┴────────────────────────┴────────────────────────────

## Stay Updated[¶][57]

Subscribe to our newsletter for updates on new features and usage tips:

Looking for more structured learning? Check out our [Tutorial series][58] for step-by-step guides.

Was this page helpful?
Thanks for your feedback!
Thanks for your feedback! Help us improve this page by using our [feedback form][59].

[1]: #instructor-cookbooks
[2]: #text-processing
[3]: #multi-modal-examples
[4]: #data-tools
[5]: #deployment-options
[6]: #text-processing
[7]: #classification-examples
[8]: single_classification/
[9]: multiple_classification/
[10]: classification/
[11]: bulk_classification/
[12]: batch_classification_langsmith/
[13]: local_classification/
[14]: #information-extraction
[15]: entity_resolution/
[16]: extract_contact_info/
[17]: pii/
[18]: exact_citations/
[19]: action_items/
[20]: search/
[21]: #document-processing
[22]: document_segmentation/
[23]: planning-tasks/
[24]: knowledge_graph/
[25]: building_knowledge_graphs/
[26]: ../tutorials/6-chain-of-density.ipynb
[27]: #multi-modal-examples
[28]: #vision-processing
[29]: tables_from_vision/
[30]: extracting_tables/
[31]: extracting_receipts/
[32]: extract_slides/
[33]: image_to_ad_copy/
[34]: youtube_clips/
[35]: #multi-modal-processing
[36]: multi_modal_gemini/
[37]: #data-tools
[38]: #database-integration
[39]: sqlmodel/
[40]: pandas_df/
[41]: #streaming-and-processing
[42]: partial_streaming/
[43]: self_critique/
[44]: #api-integration
[45]: moderation/
[46]: batch_job_oai/
[47]: examples/
[48]: #observability-tracing
[49]: tracing_with_langfuse/
[50]: #deployment-options
[51]: #model-providers
[52]: groq/
[53]: mistral/
[54]: watsonx/
[55]: #local-deployment
[56]: ollama/
[57]: #stay-updated
[58]: ../tutorials/
[59]: https://forms.gle/ijr9Zrcg2QWgKoWs7
