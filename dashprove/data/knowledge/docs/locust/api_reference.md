* API

# API[][1]

## User class[][2]

* *class *User(*environment*)[[source]][3][][4]*
  Represents a “user” which is to be spawned and attack the system that is to be load tested.
  
  The behaviour of this user is defined by its tasks. Tasks can be declared either directly on the
  class by using the [`@task decorator`][5] on methods, or by setting the [`tasks attribute`][6].
  
  This class should usually be subclassed by a class that defines some kind of client. For example
  when load testing an HTTP system, you probably want to use the [`HttpUser`][7] class.
  
  * abstract* = True*[][8]*
    If abstract is True, the class is meant to be subclassed, and locust will not spawn users of
    this class during a test.
  
  * context()[[source]][9][][10]*
    Adds the returned value (a dict) to the context for [request event][11]. Override this in your
    User class to customize the context.
  
  * environment[][12]*
    A reference to the [`Environment`][13] in which this user is running
  
  * fixed_count* = 0*[][14]*
    If the value > 0, the weight property will be ignored and the ‘fixed_count’-instances will be
    spawned. These Users are spawned first. If the total target count (specified by the –users arg)
    is not enough to spawn all instances of each User class with the defined property, the final
    count of each User is undefined.
  
  * on_start()[[source]][15][][16]*
    Called when a User starts running.
  
  * on_stop()[[source]][17][][18]*
    Called when a User stops running (is killed)
  
  * tasks* = []*[][19]*
    Collection of python callables and/or TaskSet classes that the Locust user(s) will run.
    
    If tasks is a list, the task to be performed will be picked randomly.
    
    If tasks is a *(callable,int)* list of two-tuples, or a {callable:int} dict, the task to be
    performed will be picked randomly, but each task will be weighted according to its corresponding
    int value. So in the following case, *ThreadPage* will be fifteen times more likely to be picked
    than *write_post*:
    
    class ForumPage(TaskSet):
        tasks = {ThreadPage:15, write_post:1}
  
  * wait()[[source]][20][][21]*
    Make the running user sleep for a duration defined by the User.wait_time function.
    
    The user can also be killed gracefully while it’s sleeping, so calling this method within a task
    makes it possible for a user to be killed mid-task even if you’ve set a stop_timeout. If this
    behaviour is not desired, you should make the user wait using gevent.sleep() instead.
  
  * wait_time()[][22]*
    Method that returns the time (in seconds) between the execution of locust tasks. Can be
    overridden for individual TaskSets.
    
    Example:
    
    from locust import User, between
    class MyUser(User):
        wait_time = between(3, 25)
  
  * weight* = 1*[][23]*
    Probability of user class being chosen. The higher the weight, the greater the chance of it
    being chosen.

## HttpUser class[][24]

* *class *HttpUser(**args*, ***kwargs*)[[source]][25][][26]*
  Represents an HTTP “user” which is to be spawned and attack the system that is to be load tested.
  
  The behaviour of this user is defined by its tasks. Tasks can be declared either directly on the
  class by using the [`@task decorator`][27] on methods, or by setting the [`tasks attribute`][28].
  
  This class creates a *client* attribute on instantiation which is an HTTP client with support for
  keeping a user session between requests.
  
  * abstract* = True*[][29]*
    If abstract is True, the class is meant to be subclassed, and users will not choose this locust
    during a test
  
  * client[][30]*
    Instance of HttpSession that is created upon instantiation of Locust. The client supports
    cookies, and therefore keeps the session between HTTP requests.
  
  * tasks* = []*[][31]*
    Collection of python callables and/or TaskSet classes that the Locust user(s) will run.
    
    If tasks is a list, the task to be performed will be picked randomly.
    
    If tasks is a *(callable,int)* list of two-tuples, or a {callable:int} dict, the task to be
    performed will be picked randomly, but each task will be weighted according to its corresponding
    int value. So in the following case, *ThreadPage* will be fifteen times more likely to be picked
    than *write_post*:
    
    class ForumPage(TaskSet):
        tasks = {ThreadPage:15, write_post:1}
  
  * wait_time()[][32]*

## HttpSession class[][33]

* *class *HttpSession(*base_url*, *request_event*, *user*, **args*, *pool_manager=None*,
***kwargs*)[[source]][34][][35]*
  Class for performing web requests and holding (session-) cookies between requests (in order to be
  able to log in and out of websites). Each request is logged so that locust can display statistics.
  
  This is a slightly extended version of [python-request][36]’s [`requests.Session`][37] class and
  mostly this class works exactly the same. However the methods for making requests (get, post,
  delete, put, head, options, patch, request) can now take a *url* argument that’s only the path
  part of the URL, in which case the host part of the URL will be prepended with the
  HttpSession.base_url which is normally inherited from a User class’ host attribute.
  
  Each of the methods for making requests also takes two additional optional arguments which are
  Locust specific and doesn’t exist in python-requests. These are:
  
  *Parameters:*
    * **name** – (optional) An argument that can be specified to use as label in Locust’s statistics
      instead of the URL path. This can be used to group different URL’s that are requested into a
      single entry in Locust’s statistics.
    * **catch_response** – (optional) Boolean argument that, if set, can be used to make a request
      return a context manager to work as argument to a with statement. This will allow the request
      to be marked as a fail based on the content of the response, even if the response code is ok
      (2xx). The opposite also works, one can use catch_response to catch a request and then mark it
      as successful even if the response code was not (i.e 500 or 404).
  
  * __init__(*base_url*, *request_event*, *user*, **args*, *pool_manager=None*,
  ***kwargs*)[[source]][38][][39]*
  
  * delete(*url*, ***, *data=None*, *json=None*, ***kwargs*)[[source]][40][][41]*
    Sends a DELETE request
  
  * get(*url*, ***, *data=None*, *json=None*, ***kwargs*)[[source]][42][][43]*
    Sends a GET request
  
  * head(*url*, ***, *data=None*, *json=None*, ***kwargs*)[[source]][44][][45]*
    Sends a HEAD request
  
  * options(*url*, ***, *data=None*, *json=None*, ***kwargs*)[[source]][46][][47]*
    Sends a OPTIONS request
  
  * patch(*url*, *data=None*, ***, *json=None*, ***kwargs*)[[source]][48][][49]*
    Sends a PATCH request
  
  * post(*url*, *data=None*, *json=None*, ***kwargs*)[[source]][50][][51]*
    Sends a POST request
  
  * put(*url*, *data=None*, ***, *json=None*, ***kwargs*)[[source]][52][][53]*
    Sends a PUT request
  
  * request(*method*, *url*, *name=None*, *catch_response=False*, *context={}*, ***, *data=None*,
  *json=None*, ***kwargs*)[[source]][54][][55]*
    Constructs and sends a [`requests.Request`][56]. Returns [`requests.Response`][57] object.
    
    *Parameters:*
      * **method** – method for the new `Request` object.
      * **url** – URL for the new `Request` object.
      * **name** – (optional) An argument that can be specified to use as label in Locust’s
        statistics instead of the URL path. This can be used to group different URL’s that are
        requested into a single entry in Locust’s statistics.
      * **catch_response** – (optional) Boolean argument that, if set, can be used to make a request
        return a context manager to work as argument to a with statement. This will allow the
        request to be marked as a fail based on the content of the response, even if the response
        code is ok (2xx). The opposite also works, one can use catch_response to catch a request and
        then mark it as successful even if the response code was not (i.e 500 or 404).
      * **params** – (optional) Dictionary or bytes to be sent in the query string for the
        `Request`.
      * **data** – (optional) Dictionary, list of tuples, bytes, or file-like object to send in the
        body of the `Request`.
      * **json** – (optional) json to send in the body of the `Request`.
      * **headers** – (optional) Dictionary of HTTP Headers to send with the `Request`.
      * **cookies** – (optional) Dict or CookieJar object to send with the `Request`.
      * **files** – (optional) Dictionary of `'filename': file-like-objects` for multipart encoding
        upload.
      * **auth** – (optional) Auth tuple or callable to enable Basic/Digest/Custom HTTP Auth.
      * **timeout** (*float** or **tuple*) – (optional) How long to wait for the server to send data
        before giving up, as a float, or a [(connect timeout, read timeout)][58] tuple.
      * **allow_redirects** (*bool*) – (optional) Set to True by default.
      * **proxies** – (optional) Dictionary mapping protocol or protocol and hostname to the URL of
        the proxy.
      * **hooks** – (optional) Dictionary mapping hook name to one event or list of events, event
        must be callable.
      * **stream** – (optional) whether to immediately download the response content. Defaults to
        `False`.
      * **verify** – (optional) Either a boolean, in which case it controls whether we verify the
        server’s TLS certificate, or a string, in which case it must be a path to a CA bundle to
        use. Defaults to `True`. When set to `False`, requests will accept any TLS certificate
        presented by the server, and will ignore hostname mismatches and/or expired certificates,
        which will make your application vulnerable to man-in-the-middle (MitM) attacks. Setting
        verify to `False` may be useful during local development or testing.
      * **cert** – (optional) if String, path to ssl client cert file (.pem). If Tuple, (‘cert’,
        ‘key’) pair.

## FastHttpUser class[][59]

* *class *FastHttpUser(*environment*)[[source]][60]*
  FastHttpUser provides the same API as HttpUser, but uses geventhttpclient instead of
  python-requests as its underlying client. It uses considerably less CPU on the load generator, and
  should work as a simple drop-in-replacement in most cases.
  
  * abstract* = True**
    Dont register this as a User class that can be run by itself
  
  * client*
    Instance of FastHttpSession that is created upon instantiation of User. The client support
    cookies, and therefore keeps the session between HTTP requests.
  
  * rest(*method*, *url*, *headers=None*, ***kwargs*)[[source]][61]*
    A wrapper for self.client.request that:
    
    * Parses the JSON response to a dict called `js` in the response object. Marks the request as
      failed if the response was not valid JSON.
    * Defaults `Content-Type` and `Accept` headers to `application/json`
    * Sets `catch_response=True` (so always use a [with-block][62])
    * Catches any unhandled exceptions thrown inside your with-block, marking the sample as failed
      (instead of exiting the task immediately without even firing the request event)
  
  * tasks* = []**
    Collection of python callables and/or TaskSet classes that the Locust user(s) will run.
    
    If tasks is a list, the task to be performed will be picked randomly.
    
    If tasks is a *(callable,int)* list of two-tuples, or a {callable:int} dict, the task to be
    performed will be picked randomly, but each task will be weighted according to its corresponding
    int value. So in the following case, *ThreadPage* will be fifteen times more likely to be picked
    than *write_post*:
    
    class ForumPage(TaskSet):
        tasks = {ThreadPage:15, write_post:1}
  
  * wait_time()*

## MqttUser class[][63]

## SocketIOUser class[][64]

* *class *SocketIOUser(**args*, ***kwargs*)[[source]][65]*
  SocketIOUser creates an instance of [`socketio.Client`][66] to log requests. See example in
  [examples/socketio/socketio_ex.py][67].
  
  * options* = {}**
    socketio.Client options, e.g. {“reconnection_attempts”: 1, “reconnection_delay”: 2, “logger”:
    True, “engineio_logger”: True}

## FastHttpSession class[][68]

* *class *FastHttpSession(*base_url*, *request_event*, *user*, *insecure=True*, *client_pool=None*,
*ssl_context_factory=None*, ***kwargs*)[[source]][69][][70]*
  * __init__(*base_url*, *request_event*, *user*, *insecure=True*, *client_pool=None*,
  *ssl_context_factory=None*, ***kwargs*)[[source]][71][][72]*
  
  * delete(*url*, ***kwargs*)[[source]][73][][74]*
    Sends a DELETE request
  
  * get(*url*, ***kwargs*)[[source]][75][][76]*
    Sends a GET request
  
  * head(*url*, ***kwargs*)[[source]][77][][78]*
    Sends a HEAD request
  
  * iter_lines(*url*, *method='GET'*, ***kwargs*)[[source]][79][][80]*
    Sends a iter_lines request
  
  * options(*url*, ***kwargs*)[[source]][81][][82]*
    Sends a OPTIONS request
  
  * patch(*url*, *data=None*, ***kwargs*)[[source]][83][][84]*
    Sends a PATCH request
  
  * post(*url*, *data=None*, *json=None*, ***kwargs*)[[source]][85][][86]*
    Sends a POST request
  
  * put(*url*, *data=None*, ***kwargs*)[[source]][87][][88]*
    Sends a PUT request
  
  * request(*method*, *url*, *name=None*, *data=None*, *catch_response=False*, *stream=False*,
  *headers=None*, *auth=None*, *json=None*, *allow_redirects=True*, *context={}*,
  ***kwargs*)[[source]][89][][90]*
    Send an HTTP request
    
    *Parameters:*
      * **method** – method for the new `Request` object.
      * **url** – path that will be concatenated with the base host URL that has been specified. Can
        also be a full URL, in which case the full URL will be requested, and the base host is
        ignored.
      * **name** – (optional) An argument that can be specified to use as label in Locust’s
        statistics instead of the URL path. This can be used to group different URL’s that are
        requested into a single entry in Locust’s statistics.
      * **catch_response** – (optional) Boolean argument that, if set, can be used to make a request
        return a context manager to work as argument to a with statement. This will allow the
        request to be marked as a fail based on the content of the response, even if the response
        code is ok (2xx). The opposite also works, one can use catch_response to catch a request and
        then mark it as successful even if the response code was not (i.e. 500 or 404).
      * **data** – (optional) String/bytes to send in the body of the request.
      * **json** – (optional) Json to send in the body of the request. Automatically sets
        Content-Type and Accept headers to “application/json”. Only used if data is not set.
      * **headers** – (optional) Dictionary of HTTP Headers to send with the request.
      * **auth** – (optional) Auth (username, password) tuple to enable Basic HTTP Auth.
      * **stream** – (optional) If set to true the response body will not be consumed immediately
        and can instead be consumed by accessing the stream attribute on the Response object.
        Another side effect of setting stream to True is that the time for downloading the response
        content will not be accounted for in the request time that is reported by Locust.
      * **allow_redirects** – (optional) Set to True by default.
    *Returns:*
      A [`FastResponse`][91] object if catch_response is False, and `ResponseContextManager` if
      True.

## PostgresUser class[][92]

* *class *PostgresUser(**args*, ***kwargs*)[[source]][93]*
  * abstract* = True**
    If abstract is True, the class is meant to be subclassed, and locust will not spawn users of
    this class during a test.
  
  * tasks* = []**
    Collection of python callables and/or TaskSet classes that the Locust user(s) will run.
    
    If tasks is a list, the task to be performed will be picked randomly.
    
    If tasks is a *(callable,int)* list of two-tuples, or a {callable:int} dict, the task to be
    performed will be picked randomly, but each task will be weighted according to its corresponding
    int value. So in the following case, *ThreadPage* will be fifteen times more likely to be picked
    than *write_post*:
    
    class ForumPage(TaskSet):
        tasks = {ThreadPage:15, write_post:1}
  
  * wait_time()*

## MongoDBUser class[][94]

* *class *MongoDBUser(**args*, ***kwargs*)[[source]][95]*
  * abstract* = True**
    If abstract is True, the class is meant to be subclassed, and locust will not spawn users of
    this class during a test.
  
  * tasks* = []**
    Collection of python callables and/or TaskSet classes that the Locust user(s) will run.
    
    If tasks is a list, the task to be performed will be picked randomly.
    
    If tasks is a *(callable,int)* list of two-tuples, or a {callable:int} dict, the task to be
    performed will be picked randomly, but each task will be weighted according to its corresponding
    int value. So in the following case, *ThreadPage* will be fifteen times more likely to be picked
    than *write_post*:
    
    class ForumPage(TaskSet):
        tasks = {ThreadPage:15, write_post:1}
  
  * wait_time()*

## MilvusUser class[][96]

* *class *MilvusUser(*environment*, *uri='http://localhost:19530'*, *token='root:Milvus'*,
*collection_name='test_collection'*, *db_name='default'*, *timeout=60*, *schema=None*,
*index_params=None*, ***kwargs*)[[source]][97]*
  Locust User implementation for Milvus operations.
  
  This class wraps the MilvusV2Client implementation and translates client method results into
  Locust request events so that performance statistics are collected properly.
  
  ### Parameters[][98]
  
  *hoststr*
    Milvus server URI, e.g. `"http://localhost:19530"`.
  *collection_namestr*
    The name of the collection to operate on.
  *[**][99]client_kwargs*
    Additional keyword arguments forwarded to the client.
  
  * abstract* = True**
    If abstract is True, the class is meant to be subclassed, and locust will not spawn users of
    this class during a test.
  
  * tasks* = []**
    Collection of python callables and/or TaskSet classes that the Locust user(s) will run.
    
    If tasks is a list, the task to be performed will be picked randomly.
    
    If tasks is a *(callable,int)* list of two-tuples, or a {callable:int} dict, the task to be
    performed will be picked randomly, but each task will be weighted according to its corresponding
    int value. So in the following case, *ThreadPage* will be fifteen times more likely to be picked
    than *write_post*:
    
    class ForumPage(TaskSet):
        tasks = {ThreadPage:15, write_post:1}
  
  * wait_time()*

## DNSUser class[][100]

* *class *DNSUser(*environment*)[[source]][101]*
  DNSUser provides a locust client class for dnspython’s [`dns.query`][102] methods. See example in
  [examples/dns_ex.py][103].
  
  * client*
    Example (inside task method):
    
    message = dns.message.make_query("example.com", dns.rdatatype.A)
    self.client.udp(message, "1.1.1.1")
    self.client.https(message, "1.1.1.1")

## TaskSet class[][104]

* *class *TaskSet(*parent*)[[source]][105][][106]*
  Class defining a set of tasks that a User will execute.
  
  When a TaskSet starts running, it will pick a task from the *tasks* attribute, execute it, and
  then sleep for the number of seconds returned by its *wait_time* function. If no wait_time method
  has been declared on the TaskSet, it’ll call the wait_time function on the User by default. It
  will then schedule another task for execution and so on.
  
  TaskSets can be nested, which means that a TaskSet’s *tasks* attribute can contain another
  TaskSet. If the nested TaskSet is scheduled to be executed, it will be instantiated and called
  from the currently executing TaskSet. Execution in the currently running TaskSet will then be
  handed over to the nested TaskSet which will continue to run until it throws an InterruptTaskSet
  exception, which is done when [`TaskSet.interrupt()`][107] is called. (execution will then
  continue in the first TaskSet).
  
  * *property *client[][108]*
    Shortcut to the client `client` attribute of this TaskSet’s [`User`][109]
  
  * interrupt(*reschedule=True*)[[source]][110][][111]*
    Interrupt the TaskSet and hand over execution control back to the parent TaskSet.
    
    If *reschedule* is True (default), the parent User will immediately re-schedule, and execute, a
    new task.
  
  * on_start()[[source]][112][][113]*
    Called when a User starts executing this TaskSet
  
  * on_stop()[[source]][114][][115]*
    Called when a User stops executing this TaskSet. E.g. when TaskSet.interrupt() is called or when
    the User is killed
  
  * *property *parent[][116]*
    Parent TaskSet instance of this TaskSet (or [`User`][117] if this is not a nested TaskSet)
  
  * schedule_task(*task_callable*, *first=False*)[[source]][118][][119]*
    Add a task to the User’s task execution queue.
    
    *Parameters:*
      * **task_callable** – User task to schedule.
      * **first** – Optional keyword argument. If True, the task will be put first in the queue.
  
  * tasks* = []*[][120]*
    Collection of python callables and/or TaskSet classes that the User(s) will run.
    
    If tasks is a list, the task to be performed will be picked randomly.
    
    If tasks is a *(callable,int)* list of two-tuples, or a {callable:int} dict, the task to be
    performed will be picked randomly, but each task will be weighted according to its corresponding
    int value. So in the following case, *ThreadPage* will be fifteen times more likely to be picked
    than *write_post*:
    
    class ForumPage(TaskSet):
        tasks = {ThreadPage:15, write_post:1}
  
  * *property *user[][121]*
    [`User`][122] instance that this TaskSet was created by
  
  * wait()[[source]][123][][124]*
    Make the running user sleep for a duration defined by the Locust.wait_time function (or
    TaskSet.wait_time function if it’s been defined).
    
    The user can also be killed gracefully while it’s sleeping, so calling this method within a task
    makes it possible for a user to be killed mid-task, even if you’ve set a stop_timeout. If this
    behaviour is not desired you should make the user wait using gevent.sleep() instead.
  
  * wait_time()[[source]][125][][126]*
    Method that returns the time (in seconds) between the execution of tasks.
    
    Example:
    
    from locust import TaskSet, between
    class Tasks(TaskSet):
        wait_time = between(3, 25)

## task decorator[][127]

* task(*weight=1*)[[source]][128][][129]*
  Used as a convenience decorator to be able to declare tasks for a User or a TaskSet inline in the
  class. Example:
  
  class ForumPage(TaskSet):
      @task(100)
      def read_thread(self):
          pass
  
      @task(7)
      def create_thread(self):
          pass
  
      @task(25)
      class ForumThread(TaskSet):
          @task
          def get_author(self):
              pass
  
          @task
          def get_created(self):
              pass

## tag decorator[][130]

* tag(**tags*)[[source]][131][][132]*
  Decorator for tagging tasks and TaskSets with the given tag name. You can then limit the test to
  only execute tasks that are tagged with any of the tags provided by the `--tags` command-line
  argument. Example:
  
  class ForumPage(TaskSet):
      @tag('thread')
      @task(100)
      def read_thread(self):
          pass
  
      @tag('thread')
      @tag('post')
      @task(7)
      def create_thread(self):
          pass
  
      @tag('post')
      @task(11)
      def comment(self):
          pass

## SequentialTaskSet class[][133]

* *class *SequentialTaskSet(**args*, ***kwargs*)[[source]][134][][135]*
  Class defining a sequence of tasks that a User will execute.
  
  Works like TaskSet, but task weight is ignored, and all tasks are executed in order. Tasks can
  either be specified by setting the *tasks* attribute to a list of tasks, or by declaring tasks as
  methods using the @task decorator. The order of declaration decides the order of execution.
  
  It’s possible to combine a task list in the *tasks* attribute, with some tasks declared using the
  @task decorator. The order of declaration is respected also in that case.
  
  * *property *client[][136]*
    Shortcut to the client `client` attribute of this TaskSet’s [`User`][137]
  
  * interrupt(*reschedule=True*)[][138]*
    Interrupt the TaskSet and hand over execution control back to the parent TaskSet.
    
    If *reschedule* is True (default), the parent User will immediately re-schedule, and execute, a
    new task.
  
  * on_start()[][139]*
    Called when a User starts executing this TaskSet
  
  * on_stop()[][140]*
    Called when a User stops executing this TaskSet. E.g. when TaskSet.interrupt() is called or when
    the User is killed
  
  * *property *parent[][141]*
    Parent TaskSet instance of this TaskSet (or [`User`][142] if this is not a nested TaskSet)
  
  * schedule_task(*task_callable*, *first=False*)[][143]*
    Add a task to the User’s task execution queue.
    
    *Parameters:*
      * **task_callable** – User task to schedule.
      * **first** – Optional keyword argument. If True, the task will be put first in the queue.
  
  * tasks* = []*[][144]*
    Collection of python callables and/or TaskSet classes that the User(s) will run.
    
    If tasks is a list, the task to be performed will be picked randomly.
    
    If tasks is a *(callable,int)* list of two-tuples, or a {callable:int} dict, the task to be
    performed will be picked randomly, but each task will be weighted according to its corresponding
    int value. So in the following case, *ThreadPage* will be fifteen times more likely to be picked
    than *write_post*:
    
    class ForumPage(TaskSet):
        tasks = {ThreadPage:15, write_post:1}
  
  * *property *user[][145]*
    [`User`][146] instance that this TaskSet was created by
  
  * wait_time()[][147]*
    Method that returns the time (in seconds) between the execution of tasks.
    
    Example:
    
    from locust import TaskSet, between
    class Tasks(TaskSet):
        wait_time = between(3, 25)

## Built in wait_time functions[][148]

* between(*min_wait*, *max_wait*)[][149]*
  Returns a function that will return a random number between min_wait and max_wait.
  
  Example:
  
  class MyUser(User):
      # wait between 3.0 and 10.5 seconds after each task
      wait_time = between(3.0, 10.5)

* constant(*wait_time*)[][150]*
  Returns a function that just returns the number specified by the wait_time argument
  
  Example:
  
  class MyUser(User):
      wait_time = constant(3)

* constant_pacing(*wait_time*)[][151]*
  Returns a function that will track the run time of the tasks, and for each time it’s called it
  will return a wait time that will try to make the total time between task execution equal to the
  time specified by the wait_time argument.
  
  In the following example the task will always be executed once every 10 seconds, no matter the
  task execution time:
  
  class MyUser(User):
      wait_time = constant_pacing(10)
      @task
      def my_task(self):
          time.sleep(random.random())
  
  If a task execution exceeds the specified wait_time, the wait will be 0 before starting the next
  task.

* constant_throughput(*task_runs_per_second*)[][152]*
  Returns a function that will track the run time of the tasks, and for each time it’s called it
  will return a wait time that will try to make the number of task runs per second execution equal
  to the time specified by the task_runs_per_second argument.
  
  If you have multiple requests in a task your RPS will of course be higher than the specified
  throughput.
  
  This is the mathematical inverse of constant_pacing.
  
  In the following example the task will always be executed once every 10 seconds, no matter the
  task execution time:
  
  class MyUser(User):
      wait_time = constant_throughput(0.1)
      @task
      def my_task(self):
          time.sleep(random.random())
  
  If a task execution exceeds the specified wait_time, the wait will be 0 before starting the next
  task.

## Response class[][153]

This class actually resides in the [requests][154] library, since that’s what Locust is using to
make HTTP requests, but it’s included in the API docs for locust since it’s so central when writing
locust load tests. You can also look at the [`Response`][155] class at the [requests
documentation][156].

* *class *Response[[source]][157]*
  The `Response` object, which contains a server’s response to an HTTP request.
  
  * *property *apparent_encoding*
    The apparent encoding, provided by the charset_normalizer or chardet libraries.
  
  * close()[[source]][158]*
    Releases the connection back to the pool. Once this method has been called the underlying `raw`
    object must not be accessed again.
    
    *Note: Should not normally need to be called explicitly.*
  
  * *property *content*
    Content of the response, in bytes.
  
  * cookies*
    A CookieJar of Cookies the server sent back.
  
  * elapsed*
    The amount of time elapsed between sending the request and the arrival of the response (as a
    timedelta). This property specifically measures the time taken between sending the first byte of
    the request and finishing parsing the headers. It is therefore unaffected by consuming the
    response content or the value of the `stream` keyword argument.
  
  * encoding*
    Encoding to decode with when accessing r.text.
  
  * headers*
    Case-insensitive Dictionary of Response Headers. For example, `headers['content-encoding']` will
    return the value of a `'Content-Encoding'` response header.
  
  * history*
    A list of `Response` objects from the history of the Request. Any redirect responses will end up
    here. The list is sorted from the oldest to the most recent request.
  
  * *property *is_permanent_redirect*
    True if this Response one of the permanent versions of redirect.
  
  * *property *is_redirect*
    True if this Response is a well-formed HTTP redirect that could have been processed
    automatically (by `Session.resolve_redirects()`).
  
  * iter_content(*chunk_size=1*, *decode_unicode=False*)[[source]][159]*
    Iterates over the response data. When stream=True is set on the request, this avoids reading the
    content at once into memory for large responses. The chunk size is the number of bytes it should
    read into memory. This is not necessarily the length of each item returned as decoding can take
    place.
    
    chunk_size must be of type int or None. A value of None will function differently depending on
    the value of stream. stream=True will read data as it arrives in whatever size the chunks are
    received. If stream=False, data is returned as a single chunk.
    
    If decode_unicode is True, content will be decoded using the best available encoding based on
    the response.
  
  * iter_lines(*chunk_size=512*, *decode_unicode=False*, *delimiter=None*)[[source]][160]*
    Iterates over the response data, one line at a time. When stream=True is set on the request,
    this avoids reading the content at once into memory for large responses.
    
    Note
    
    This method is not reentrant safe.
  
  * json(***kwargs*)[[source]][161]*
    Decodes the JSON response body (if any) as a Python object.
    
    This may return a dictionary, list, etc. depending on what is in the response.
    
    *Parameters:*
      ****kwargs** – Optional arguments that `json.loads` takes.
    *Raises:*
      [**requests.exceptions.JSONDecodeError**][162] – If the response body does not contain valid
      json.
  
  * *property *links*
    Returns the parsed header links of the response, if any.
  
  * *property *next*
    Returns a PreparedRequest for the next request in a redirect chain, if there is one.
  
  * *property *ok*
    Returns True if [`status_code`][163] is less than 400, False if not.
    
    This attribute checks if the status code of the response is between 400 and 600 to see if there
    was a client error or a server error. If the status code is between 200 and 400, this will
    return True. This is **not** a check to see if the response code is `200 OK`.
  
  * raise_for_status()[[source]][164]*
    Raises `HTTPError`, if one occurred.
  
  * raw*
    File-like object representation of response (for advanced usage). Use of `raw` requires that
    `stream=True` be set on the request. This requirement does not apply for use internally to
    Requests.
  
  * reason*
    Textual reason of responded HTTP Status, e.g. “Not Found” or “OK”.
  
  * request*
    The `PreparedRequest` object to which this is a response.
  
  * status_code*
    Integer Code of responded HTTP Status, e.g. 404 or 200.
  
  * *property *text*
    Content of the response, in unicode.
    
    If Response.encoding is None, encoding will be guessed using `charset_normalizer` or `chardet`.
    
    The encoding of the response content is determined based solely on HTTP headers, following RFC
    2616 to the letter. If you can take advantage of non-HTTP knowledge to make a better guess at
    the encoding, you should set `r.encoding` appropriately before accessing this property.
  
  * url*
    Final URL location of Response.

## ResponseContextManager class[][165]

* *class *ResponseContextManager(*error*)[[source]][166][][167]*
  A Response class that also acts as a context manager that provides the ability to manually control
  if an HTTP request should be marked as successful or a failure in Locust’s statistics
  
  This class is a subclass of [`Response`][168] with two additional methods: [`success`][169] and
  [`failure`][170].
  
  * failure(*exc*)[[source]][171][][172]*
    Report the response as a failure.
    
    if exc is anything other than a python exception (like a string) it will be wrapped inside a
    CatchResponseError.
    
    Example:
    
    with self.client.get("/", catch_response=True) as response:
        if response.content == b"":
            response.failure("No data")
  
  * success()[[source]][173][][174]*
    Report the response as successful
    
    Example:
    
    with self.client.get("/does/not/exist", catch_response=True) as response:
        if response.status_code == 404:
            response.success()

## Exceptions[][175]

* *exception *InterruptTaskSet(*reschedule=True*)[[source]][176][][177]*
  Exception that will interrupt a User when thrown inside a task

* *exception *RescheduleTask[[source]][178][][179]*
  When raised in a task it’s equivalent of a return statement.
  
  Also used internally by TaskSet. When raised within the task control flow of a TaskSet, but not
  inside a task, the execution should be handed over to the parent TaskSet.

* *exception *RescheduleTaskImmediately[[source]][180][][181]*
  When raised in a User task, another User task will be rescheduled immediately (without calling
  wait_time first)

## Environment class[][182]

* *class *Environment(***, *user_classes=None*, *shape_class=None*, *tags=None*, *locustfile=None*,
*exclude_tags=None*, *events=None*, *host=None*, *reset_stats=False*, *stop_timeout=None*,
*catch_exceptions=True*, *parsed_options=None*, *parsed_locustfiles=None*,
*available_user_classes=None*, *available_shape_classes=None*, *available_user_tasks=None*,
*dispatcher_class=<class 'locust.dispatch.UsersDispatcher'>*,
*profile=None*)[[source]][183][][184]*
  * assign_equal_weights()[[source]][185][][186]*
    Update the user classes such that each user runs their specified tasks with equal probability.
  
  * available_shape_classes[][187]*
    List of the available Shape Classes to pick from in the ShapeClass Picker
  
  * available_user_classes[][188]*
    List of the available User Classes to pick from in the UserClass Picker
  
  * available_user_tasks[][189]*
    List of the available Tasks per User Classes to pick from in the Task Picker
  
  * catch_exceptions[][190]*
    If True exceptions that happen within running users will be caught (and reported in UI/console).
    If False, exceptions will be raised.
  
  * create_local_runner()[[source]][191][][192]*
    Create a [`LocalRunner`][193] instance for this Environment
  
  * create_master_runner(*master_bind_host='*'*, *master_bind_port=5557*)[[source]][194][][195]*
    Create a [`MasterRunner`][196] instance for this Environment
    
    *Parameters:*
      * **master_bind_host** – Interface/host that the master should use for incoming worker
        connections. Defaults to “*” which means all interfaces.
      * **master_bind_port** – Port that the master should listen for incoming worker connections on
  
  * create_web_ui(*host=''*, *port=8089*, *web_base_path=None*, *web_login=False*, *tls_cert=None*,
  *tls_key=None*, *stats_csv_writer=None*, *delayed_start=False*,
  *userclass_picker_is_active=False*, *build_path=None*)[[source]][197][][198]*
    Creates a [`WebUI`][199] instance for this Environment and start running the web server
    
    *Parameters:*
      * **host** – Host/interface that the web server should accept connections to. Defaults to “”
        which means all interfaces
      * **port** – Port that the web server should listen to
      * **web_login** – If provided, an authentication page will protect the app
      * **tls_cert** – An optional path (str) to a TLS cert. If this is provided the web UI will be
        served over HTTPS
      * **tls_key** – An optional path (str) to a TLS private key. If this is provided the web UI
        will be served over HTTPS
      * **stats_csv_writer** – StatsCSV <stats_csv.StatsCSV> instance.
      * **delayed_start** – Whether or not to delay starting web UI until start() is called.
        Delaying web UI start allows for adding Flask routes or Blueprints before accepting
        requests, avoiding errors.
  
  * create_worker_runner(*master_host*, *master_port*)[[source]][200][][201]*
    Create a [`WorkerRunner`][202] instance for this Environment
    
    *Parameters:*
      * **master_host** – Host/IP of a running master node
      * **master_port** – Port on master node to connect to
  
  * dispatcher_class[][203]*
    A user dispatcher class that decides how users are spawned, default `UsersDispatcher`
  
  * events[][204]*
    Event hooks used by Locust internally, as well as to extend Locust’s functionality See [Event
    hooks][205] for available events.
  
  * exclude_tags[][206]*
    If set, only tasks that aren’t tagged by tags in this list will be executed. Leave this as None
    to use the one from parsed_options
  
  * host[][207]*
    Base URL of the target system
  
  * locustfile[][208]*
    Filename (not path) of locustfile
  
  * parsed_locustfiles[][209]*
    A list of all locustfiles for the test
  
  * parsed_options[][210]*
    Reference to the parsed command line options (used to pre-populate fields in Web UI). When using
    Locust as a library, this should either be None or an object created by
    argument_parser.parse_args()
  
  * process_exit_code[][211]*
    If set it’ll be the exit code of the Locust process
  
  * profile[][212]*
    Profile name for the test run
  
  * reset_stats[][213]*
    Determines if stats should be reset once all simulated users have been spawned
  
  * runner[][214]*
    Reference to the [`Runner`][215] instance
  
  * shape_class[][216]*
    A shape class to control the shape of the load test
  
  * stats[][217]*
    Reference to RequestStats instance
  
  * tags[][218]*
    If set, only tasks that are tagged by tags in this list will be executed. Leave this as None to
    use the one from parsed_options
  
  * user_classes[][219]*
    User classes that the runner will run
  
  * web_ui[][220]*
    Reference to the WebUI instance
  
  * worker_logs[][221]*
    Captured logs from all connected workers

## Event hooks[][222]

Locust provides event hooks that can be used to extend Locust in various ways.

The following event hooks are available under [`Environment.events`][223], and there’s also a
reference to these events under `locust.events` that can be used at the module level of locust
scripts (since the Environment instance hasn’t been created when the locustfile is imported).

* *class *Events[[source]][224][][225]*
  * cpu_warning[][226]*
    Fired when the CPU usage exceeds runners.CPU_WARNING_THRESHOLD (90% by default)
  
  * heartbeat_received[][227]*
    Fired when a heartbeat is received by a worker from master.
    
    Event arguments:
    
    *Parameters:*
      * **client_id** – worker client id
      * **timestamp** – time in seconds since the epoch (float) when the event occured
  
  * heartbeat_sent[][228]*
    Fired when a heartbeat is sent by master to a worker.
    
    Event arguments:
    
    *Parameters:*
      * **client_id** – worker client id
      * **timestamp** – time in seconds since the epoch (float) when the event occured
  
  * init[][229]*
    Fired when Locust is started, once the Environment instance and locust runner instance have been
    created. This hook can be used by end-users’ code to run code that requires access to the
    Environment. For example to register listeners to other events.
    
    Event arguments:
    
    *Parameters:*
      **environment** – Environment instance
  
  * init_command_line_parser[][230]*
    Event that can be used to add command line options to Locust
    
    Event arguments:
    
    *Parameters:*
      **parser** – ArgumentParser instance
  
  * quit[][231]*
    Fired after quitting events, just before process is exited.
    
    Event arguments:
    
    *Parameters:*
      **exit_code** – Exit code for process
  
  * quitting[][232]*
    Fired when the locust process is exiting.
    
    Event arguments:
    
    *Parameters:*
      **environment** – Environment instance
  
  * report_to_master[][233]*
    Used when Locust is running in –worker mode. It can be used to attach data to the dicts that are
    regularly sent to the master. It’s fired regularly when a report is to be sent to the master
    server.
    
    Note that the keys “stats” and “errors” are used by Locust and shouldn’t be overridden.
    
    Event arguments:
    
    *Parameters:*
      * **client_id** – The client id of the running locust process.
      * **data** – Data dict that can be modified in order to attach data that should be sent to the
        master.
  
  * request[][234]*
    Fired when a request in completed.
    
    Event arguments:
    
    *Parameters:*
      * **request_type** – Request type method used
      * **name** – Path to the URL that was called (or override name if it was used in the call to
        the client)
      * **response_time** – Time in milliseconds until exception was thrown
      * **response_length** – Content-length of the response
      * **response** – Response object (e.g. a [`requests.Response`][235])
      * **context** – [User/request context][236]
      * **exception** – Exception instance that was thrown. None if request was successful.
    
    If you want to simplify a custom client, you can have Locust measure the time for you by using
    [`measure()`][237]
  
  * reset_stats[][238]*
    Fired when the Reset Stats button is clicked in the web UI.
  
  * spawning_complete[][239]*
    Fired when all simulated users has been spawned. The event is fired on master first, and then
    distributed to workers.
    
    Event arguments:
    
    *Parameters:*
      **user_count** – Number of users that were spawned (in total, not per-worker)
  
  * test_start[][240]*
    Fired on each node when a new load test is started. It’s not fired again if the number of users
    change during a test.
  
  * test_stop[][241]*
    Fired on each node when a load test is stopped.
  
  * test_stopping[][242]*
    Fired on each node when a load test is about to stop - before stopping users.
  
  * usage_monitor[][243]*
    Fired every runners.CPU_MONITOR_INTERVAL (5.0 seconds by default) with information about current
    CPU and memory usage.
    
    Event arguments:
    
    *Parameters:*
      * **environment** – locust environment
      * **cpu_usage** – current CPU usage in percent
      * **memory_usage** – current memory usage (RSS) in bytes
  
  * user_error[][244]*
    Fired when an exception occurs inside the execution of a User class.
    
    Event arguments:
    
    *Parameters:*
      * **user_instance** – User class instance where the exception occurred
      * **exception** – Exception that was thrown
      * **tb** – Traceback object (from e.__traceback__)
  
  * worker_connect[][245]*
    Fired on master when a new worker connects. Note that is fired immediately after the connection
    is established, so init event may not yet have finished on worker.
    
    *Parameters:*
      **client_id** – Client id of the connected worker
  
  * worker_report[][246]*
    Used when Locust is running in –master mode and is fired when the master server receives a
    report from a Locust worker server.
    
    This event can be used to aggregate data from the locust worker servers.
    
    Event arguments:
    
    *Parameters:*
      * **client_id** – Client id of the reporting worker
      * **data** – Data dict with the data from the worker node

Note

It’s highly recommended that you add a wildcard keyword argument in your event listeners to prevent
your code from breaking if new arguments are added in a future version.

### EventHook class[][247]

The event hooks are instances of the **locust.events.EventHook** class:

* *class *EventHook[[source]][248][][249]*
  Simple event class used to provide hooks for different types of events in Locust.
  
  Here’s how to use the EventHook class:
  
  my_event = EventHook()
  def on_my_event(a, b, **kw):
      print("Event was fired with arguments: %s, %s" % (a, b))
  my_event.add_listener(on_my_event)
  my_event.fire(a="foo", b="bar")
  
  If reverse is True, then the handlers will run in the reverse order that they were inserted
  
  * measure(*request_type*, *name*, *response_length=0*, *context=None*)[[source]][250][][251]*
    Convenience method for firing the event with automatically calculated response time and
    automatically marking the request as failed if an exception is raised (this is really only
    useful for the *request* event)
    
    Example usage (in a task):
    
    with self.environment.events.request.measure("requestType", "requestName") as request_meta:
        # do the stuff you want to measure
    
    You can optionally add/overwrite entries in the request_meta dict and they will be passed to the
    request event.
    
    Experimental.

## Runner classes[][252]

* *class *Runner(*environment*)[[source]][253][][254]*
  Orchestrates the load test by starting and stopping the users.
  
  Use one of the [`create_local_runner`][255], [`create_master_runner`][256] or
  [`create_worker_runner`][257] methods on the [`Environment`][258] instance to create a runner of
  the desired type.
  
  * quit()[[source]][259][][260]*
    Stop any running load test and kill all greenlets for the runner
  
  * stop()[[source]][261][][262]*
    Stop a running load test by stopping all running users
  
  * *property *user_count[][263]*
    *Returns:*
      Number of currently running users

* *class *LocalRunner(*environment*)[[source]][264][][265]*
  Runner for running single process load test

* *class *MasterRunner(*environment*, *master_bind_host*,
*master_bind_port*)[[source]][266][][267]*
  Runner used to run distributed load tests across multiple processes and/or machines.
  
  MasterRunner doesn’t spawn any user greenlets itself. Instead it expects [`WorkerRunners`][268] to
  connect to it, which it will then direct to start and stop user greenlets. Stats sent back from
  the [`WorkerRunners`][269] will aggregated.
  
  * register_message(*msg_type*, *listener*, *concurrent=False*)[][270]*
    Register a listener for a custom message from another node
    
    *Parameters:*
      * **msg_type** – The type of the message to listen for
      * **listener** – The function to execute when the message is received
  
  * send_message(*msg_type*, *data=None*, *client_id=None*)[[source]][271][][272]*
    Sends a message to attached worker node(s)
    
    *Parameters:*
      * **msg_type** – The type of the message to send
      * **data** – Optional data to send
      * **client_id** – Optional id of the target worker node. If None, will send to all attached
        workers

* *class *WorkerRunner(*environment*, *master_host*, *master_port*)[[source]][273][][274]*
  Runner used to run distributed load tests across multiple processes and/or machines.
  
  WorkerRunner connects to a [`MasterRunner`][275] from which it’ll receive instructions to start
  and stop user greenlets. The WorkerRunner will periodically take the stats generated by the
  running users and send back to the [`MasterRunner`][276].
  
  * register_message(*msg_type*, *listener*, *concurrent=False*)[][277]*
    Register a listener for a custom message from another node
    
    *Parameters:*
      * **msg_type** – The type of the message to listen for
      * **listener** – The function to execute when the message is received
  
  * send_message(*msg_type*, *data=None*, *client_id=None*)[[source]][278][][279]*
    Sends a message to master node
    
    *Parameters:*
      * **msg_type** – The type of the message to send
      * **data** – Optional data to send
      * **client_id** – (unused)

## Web UI class[][280]

* *class *WebUI(*environment*, *host*, *port*, *web_base_path=None*, *web_login=False*,
*tls_cert=None*, *tls_key=None*, *stats_csv_writer=None*, *delayed_start=False*,
*userclass_picker_is_active=False*, *build_path=None*)[[source]][281][][282]*
  Sets up and runs a Flask web app that can start and stop load tests using the
  [`environment.runner`][283] as well as show the load test statistics in [`environment.stats`][284]
  
  * app* = None*[][285]*
    Reference to the `flask.Flask` app. Can be used to add additional web routes and customize the
    Flask app in other various ways. Example:
    
    from flask import request
    
    @web_ui.app.route("/my_custom_route")
    def my_custom_route():
        return "your IP is: %s" % request.remote_addr
  
  * auth_args[][286]*
    Arguments used to render auth.html for the web UI auth page. Must be used when configuring auth
  
  * auth_required_if_enabled(*view_func*)[[source]][287][][288]*
    Decorator that can be used on custom route methods that will turn on Flask Login authentication
    if the `--web-login` flag is used. Example:
    
    @web_ui.app.route("/my_custom_route")
    @web_ui.auth_required_if_enabled
    def my_custom_route():
        return "custom response"
  
  * greenlet* = None*[][289]*
    Greenlet of the running web server
  
  * server* = None*[][290]*
    Reference to the `pyqsgi.WSGIServer` instance
  
  * stop()[[source]][291][][292]*
    Stop the running web server
  
  * template_args[][293]*
    Arguments used to render index.html for the web UI. Must be used with custom templates extending
    index.html.

## Other[][294]

* *class *LoadTestShape[[source]][295][][296]*
  Base class for custom load shapes.
  
  * get_current_user_count()[[source]][297][][298]*
    Returns current actual number of users from the runner
  
  * get_run_time()[[source]][299][][300]*
    Calculates run time in seconds of the load test
  
  * reset_time()[[source]][301][][302]*
    Resets start time back to 0
  
  * runner* = None*[][303]*
    Reference to the [`Runner`][304] instance
  
  * *abstract *tick()[[source]][305][][306]*
    Returns a tuple with 2 elements to control the running load test:
    
    > user_count – Total user count spawn_rate – Number of users to start/stop per second when
    > changing number of users user_classes – None or a List of userclasses to be spawned in it tick
    
    If None is returned then the running load test will be stopped.

* *class *RequestStats(*use_response_times_cache=True*)[[source]][307][][308]*
  Class that holds the request statistics. Accessible in a User from self.environment.stats
  
  * get(*name*, *method*)[[source]][309][][310]*
    Retrieve a StatsEntry instance by name and method

* *class *StatsEntry(*stats*, *name*, *method*,
*use_response_times_cache=False*)[[source]][311][][312]*
  Represents a single stats entry (name and method)

* run_single_user(*user_class*, *include_length=False*, *include_time=False*,
*include_context=False*, *include_payload=False*, *loglevel='WARNING'*)[[source]][313][][314]*
  Runs a single User. Useful when you want to run a debugger.
  
  It creates in a new locust [`Environment`][315] and triggers any `init` or `test_start`
  [events][316] as normal.
  
  It does **not** trigger `test_stop` or `quit` when you quit the debugger.
  
  It prints some info about every request to stdout, and you can get additional info using the
  include_* flags
  
  It also initiates logging on WARNING level (not INFO, because it could interfere with the printing
  of requests), but you can change that by passing a log level (or disabling logging entirely by
  passing None)
[ Previous][317] [Next ][318]

© Copyright 2009-2025, Carl Byström, Jonatan Heyman, Lars Holmberg.

Built with [Sphinx][319] using a [theme][320] provided by [Read the Docs][321].

[1]: #api
[2]: #user-class
[3]: _modules/locust/user/users.html#User
[4]: #locust.User
[5]: #locust.task
[6]: #locust.User.tasks
[7]: #locust.HttpUser
[8]: #locust.User.abstract
[9]: _modules/locust/user/users.html#User.context
[10]: #locust.User.context
[11]: extending-locust.html#request-context
[12]: #locust.User.environment
[13]: #locust.env.Environment
[14]: #locust.User.fixed_count
[15]: _modules/locust/user/users.html#User.on_start
[16]: #locust.User.on_start
[17]: _modules/locust/user/users.html#User.on_stop
[18]: #locust.User.on_stop
[19]: #locust.User.tasks
[20]: _modules/locust/user/users.html#User.wait
[21]: #locust.User.wait
[22]: #locust.User.wait_time
[23]: #locust.User.weight
[24]: #httpuser-class
[25]: _modules/locust/user/users.html#HttpUser
[26]: #locust.HttpUser
[27]: #locust.task
[28]: #locust.User.tasks
[29]: #locust.HttpUser.abstract
[30]: #locust.HttpUser.client
[31]: #locust.HttpUser.tasks
[32]: #locust.HttpUser.wait_time
[33]: #httpsession-class
[34]: _modules/locust/clients.html#HttpSession
[35]: #locust.clients.HttpSession
[36]: https://requests.readthedocs.io/
[37]: https://requests.readthedocs.io/en/latest/api/#requests.Session
[38]: _modules/locust/clients.html#HttpSession.__init__
[39]: #locust.clients.HttpSession.__init__
[40]: _modules/locust/clients.html#HttpSession.delete
[41]: #locust.clients.HttpSession.delete
[42]: _modules/locust/clients.html#HttpSession.get
[43]: #locust.clients.HttpSession.get
[44]: _modules/locust/clients.html#HttpSession.head
[45]: #locust.clients.HttpSession.head
[46]: _modules/locust/clients.html#HttpSession.options
[47]: #locust.clients.HttpSession.options
[48]: _modules/locust/clients.html#HttpSession.patch
[49]: #locust.clients.HttpSession.patch
[50]: _modules/locust/clients.html#HttpSession.post
[51]: #locust.clients.HttpSession.post
[52]: _modules/locust/clients.html#HttpSession.put
[53]: #locust.clients.HttpSession.put
[54]: _modules/locust/clients.html#HttpSession.request
[55]: #locust.clients.HttpSession.request
[56]: https://requests.readthedocs.io/en/latest/api/#requests.Request
[57]: https://requests.readthedocs.io/en/latest/api/#requests.Response
[58]: https://requests.readthedocs.io/en/latest/user/advanced/#timeouts
[59]: #fasthttpuser-class
[60]: _modules/locust/contrib/fasthttp.html#FastHttpUser
[61]: _modules/locust/contrib/fasthttp.html#FastHttpUser.rest
[62]: writing-a-locustfile.html#catch-response
[63]: #mqttuser-class
[64]: #socketiouser-class
[65]: _modules/locust/contrib/socketio.html#SocketIOUser
[66]: https://python-socketio.readthedocs.io/en/stable/api.html#socketio.Client
[67]: https://github.com/locustio/locust/blob/master/examples/socketio/socketio_ex.py
[68]: #fasthttpsession-class
[69]: _modules/locust/contrib/fasthttp.html#FastHttpSession
[70]: #locust.contrib.fasthttp.FastHttpSession
[71]: _modules/locust/contrib/fasthttp.html#FastHttpSession.__init__
[72]: #locust.contrib.fasthttp.FastHttpSession.__init__
[73]: _modules/locust/contrib/fasthttp.html#FastHttpSession.delete
[74]: #locust.contrib.fasthttp.FastHttpSession.delete
[75]: _modules/locust/contrib/fasthttp.html#FastHttpSession.get
[76]: #locust.contrib.fasthttp.FastHttpSession.get
[77]: _modules/locust/contrib/fasthttp.html#FastHttpSession.head
[78]: #locust.contrib.fasthttp.FastHttpSession.head
[79]: _modules/locust/contrib/fasthttp.html#FastHttpSession.iter_lines
[80]: #locust.contrib.fasthttp.FastHttpSession.iter_lines
[81]: _modules/locust/contrib/fasthttp.html#FastHttpSession.options
[82]: #locust.contrib.fasthttp.FastHttpSession.options
[83]: _modules/locust/contrib/fasthttp.html#FastHttpSession.patch
[84]: #locust.contrib.fasthttp.FastHttpSession.patch
[85]: _modules/locust/contrib/fasthttp.html#FastHttpSession.post
[86]: #locust.contrib.fasthttp.FastHttpSession.post
[87]: _modules/locust/contrib/fasthttp.html#FastHttpSession.put
[88]: #locust.contrib.fasthttp.FastHttpSession.put
[89]: _modules/locust/contrib/fasthttp.html#FastHttpSession.request
[90]: #locust.contrib.fasthttp.FastHttpSession.request
[91]: increase-performance.html#locust.contrib.fasthttp.FastResponse
[92]: #postgresuser-class
[93]: _modules/locust/contrib/postgres.html#PostgresUser
[94]: #mongodbuser-class
[95]: _modules/locust/contrib/mongodb.html#MongoDBUser
[96]: #milvususer-class
[97]: _modules/locust/contrib/milvus.html#MilvusUser
[98]: #parameters
[99]: #id1
[100]: #dnsuser-class
[101]: _modules/locust/contrib/dns.html#DNSUser
[102]: https://dnspython.readthedocs.io/en/stable/query.html#module-dns.query
[103]: https://github.com/locustio/locust/blob/master/examples/dns_ex.py
[104]: #taskset-class
[105]: _modules/locust/user/task.html#TaskSet
[106]: #locust.TaskSet
[107]: #locust.TaskSet.interrupt
[108]: #locust.TaskSet.client
[109]: #locust.User
[110]: _modules/locust/user/task.html#TaskSet.interrupt
[111]: #locust.TaskSet.interrupt
[112]: _modules/locust/user/task.html#TaskSet.on_start
[113]: #locust.TaskSet.on_start
[114]: _modules/locust/user/task.html#TaskSet.on_stop
[115]: #locust.TaskSet.on_stop
[116]: #locust.TaskSet.parent
[117]: #locust.User
[118]: _modules/locust/user/task.html#TaskSet.schedule_task
[119]: #locust.TaskSet.schedule_task
[120]: #locust.TaskSet.tasks
[121]: #locust.TaskSet.user
[122]: #locust.User
[123]: _modules/locust/user/task.html#TaskSet.wait
[124]: #locust.TaskSet.wait
[125]: _modules/locust/user/task.html#TaskSet.wait_time
[126]: #locust.TaskSet.wait_time
[127]: #task-decorator
[128]: _modules/locust/user/task.html#task
[129]: #locust.task
[130]: #tag-decorator
[131]: _modules/locust/user/task.html#tag
[132]: #locust.tag
[133]: #sequentialtaskset-class
[134]: _modules/locust/user/sequential_taskset.html#SequentialTaskSet
[135]: #locust.SequentialTaskSet
[136]: #locust.SequentialTaskSet.client
[137]: #locust.User
[138]: #locust.SequentialTaskSet.interrupt
[139]: #locust.SequentialTaskSet.on_start
[140]: #locust.SequentialTaskSet.on_stop
[141]: #locust.SequentialTaskSet.parent
[142]: #locust.User
[143]: #locust.SequentialTaskSet.schedule_task
[144]: #locust.SequentialTaskSet.tasks
[145]: #locust.SequentialTaskSet.user
[146]: #locust.User
[147]: #locust.SequentialTaskSet.wait_time
[148]: #module-locust.wait_time
[149]: #locust.wait_time.between
[150]: #locust.wait_time.constant
[151]: #locust.wait_time.constant_pacing
[152]: #locust.wait_time.constant_throughput
[153]: #response-class
[154]: https://requests.readthedocs.io/
[155]: https://requests.readthedocs.io/en/latest/api/#requests.Response
[156]: https://requests.readthedocs.io/
[157]: _modules/requests/models.html#Response
[158]: _modules/requests/models.html#Response.close
[159]: _modules/requests/models.html#Response.iter_content
[160]: _modules/requests/models.html#Response.iter_lines
[161]: _modules/requests/models.html#Response.json
[162]: https://requests.readthedocs.io/en/latest/api/#requests.JSONDecodeError
[163]: https://requests.readthedocs.io/en/latest/api/#requests.Response.status_code
[164]: _modules/requests/models.html#Response.raise_for_status
[165]: #responsecontextmanager-class
[166]: _modules/locust/clients.html#ResponseContextManager
[167]: #locust.clients.ResponseContextManager
[168]: https://requests.readthedocs.io/en/latest/api/#requests.Response
[169]: #locust.clients.ResponseContextManager.success
[170]: #locust.clients.ResponseContextManager.failure
[171]: _modules/locust/clients.html#ResponseContextManager.failure
[172]: #locust.clients.ResponseContextManager.failure
[173]: _modules/locust/clients.html#ResponseContextManager.success
[174]: #locust.clients.ResponseContextManager.success
[175]: #exceptions
[176]: _modules/locust/exception.html#InterruptTaskSet
[177]: #locust.exception.InterruptTaskSet
[178]: _modules/locust/exception.html#RescheduleTask
[179]: #locust.exception.RescheduleTask
[180]: _modules/locust/exception.html#RescheduleTaskImmediately
[181]: #locust.exception.RescheduleTaskImmediately
[182]: #environment-class
[183]: _modules/locust/env.html#Environment
[184]: #locust.env.Environment
[185]: _modules/locust/env.html#Environment.assign_equal_weights
[186]: #locust.env.Environment.assign_equal_weights
[187]: #locust.env.Environment.available_shape_classes
[188]: #locust.env.Environment.available_user_classes
[189]: #locust.env.Environment.available_user_tasks
[190]: #locust.env.Environment.catch_exceptions
[191]: _modules/locust/env.html#Environment.create_local_runner
[192]: #locust.env.Environment.create_local_runner
[193]: #locust.runners.LocalRunner
[194]: _modules/locust/env.html#Environment.create_master_runner
[195]: #locust.env.Environment.create_master_runner
[196]: #locust.runners.MasterRunner
[197]: _modules/locust/env.html#Environment.create_web_ui
[198]: #locust.env.Environment.create_web_ui
[199]: #locust.web.WebUI
[200]: _modules/locust/env.html#Environment.create_worker_runner
[201]: #locust.env.Environment.create_worker_runner
[202]: #locust.runners.WorkerRunner
[203]: #locust.env.Environment.dispatcher_class
[204]: #locust.env.Environment.events
[205]: #events
[206]: #locust.env.Environment.exclude_tags
[207]: #locust.env.Environment.host
[208]: #locust.env.Environment.locustfile
[209]: #locust.env.Environment.parsed_locustfiles
[210]: #locust.env.Environment.parsed_options
[211]: #locust.env.Environment.process_exit_code
[212]: #locust.env.Environment.profile
[213]: #locust.env.Environment.reset_stats
[214]: #locust.env.Environment.runner
[215]: #locust.runners.Runner
[216]: #locust.env.Environment.shape_class
[217]: #locust.env.Environment.stats
[218]: #locust.env.Environment.tags
[219]: #locust.env.Environment.user_classes
[220]: #locust.env.Environment.web_ui
[221]: #locust.env.Environment.worker_logs
[222]: #event-hooks
[223]: #locust.env.Environment.events
[224]: _modules/locust/event.html#Events
[225]: #locust.event.Events
[226]: #locust.event.Events.cpu_warning
[227]: #locust.event.Events.heartbeat_received
[228]: #locust.event.Events.heartbeat_sent
[229]: #locust.event.Events.init
[230]: #locust.event.Events.init_command_line_parser
[231]: #locust.event.Events.quit
[232]: #locust.event.Events.quitting
[233]: #locust.event.Events.report_to_master
[234]: #locust.event.Events.request
[235]: https://requests.readthedocs.io/en/latest/api/#requests.Response
[236]: extending-locust.html#request-context
[237]: #locust.event.EventHook.measure
[238]: #locust.event.Events.reset_stats
[239]: #locust.event.Events.spawning_complete
[240]: #locust.event.Events.test_start
[241]: #locust.event.Events.test_stop
[242]: #locust.event.Events.test_stopping
[243]: #locust.event.Events.usage_monitor
[244]: #locust.event.Events.user_error
[245]: #locust.event.Events.worker_connect
[246]: #locust.event.Events.worker_report
[247]: #eventhook-class
[248]: _modules/locust/event.html#EventHook
[249]: #locust.event.EventHook
[250]: _modules/locust/event.html#EventHook.measure
[251]: #locust.event.EventHook.measure
[252]: #runner-classes
[253]: _modules/locust/runners.html#Runner
[254]: #locust.runners.Runner
[255]: #locust.env.Environment.create_local_runner
[256]: #locust.env.Environment.create_master_runner
[257]: #locust.env.Environment.create_worker_runner
[258]: #locust.env.Environment
[259]: _modules/locust/runners.html#Runner.quit
[260]: #locust.runners.Runner.quit
[261]: _modules/locust/runners.html#Runner.stop
[262]: #locust.runners.Runner.stop
[263]: #locust.runners.Runner.user_count
[264]: _modules/locust/runners.html#LocalRunner
[265]: #locust.runners.LocalRunner
[266]: _modules/locust/runners.html#MasterRunner
[267]: #locust.runners.MasterRunner
[268]: #locust.runners.WorkerRunner
[269]: #locust.runners.WorkerRunner
[270]: #locust.runners.MasterRunner.register_message
[271]: _modules/locust/runners.html#MasterRunner.send_message
[272]: #locust.runners.MasterRunner.send_message
[273]: _modules/locust/runners.html#WorkerRunner
[274]: #locust.runners.WorkerRunner
[275]: #locust.runners.MasterRunner
[276]: #locust.runners.MasterRunner
[277]: #locust.runners.WorkerRunner.register_message
[278]: _modules/locust/runners.html#WorkerRunner.send_message
[279]: #locust.runners.WorkerRunner.send_message
[280]: #web-ui-class
[281]: _modules/locust/web.html#WebUI
[282]: #locust.web.WebUI
[283]: #locust.env.Environment.runner
[284]: #locust.env.Environment.stats
[285]: #locust.web.WebUI.app
[286]: #locust.web.WebUI.auth_args
[287]: _modules/locust/web.html#WebUI.auth_required_if_enabled
[288]: #locust.web.WebUI.auth_required_if_enabled
[289]: #locust.web.WebUI.greenlet
[290]: #locust.web.WebUI.server
[291]: _modules/locust/web.html#WebUI.stop
[292]: #locust.web.WebUI.stop
[293]: #locust.web.WebUI.template_args
[294]: #other
[295]: _modules/locust/shape.html#LoadTestShape
[296]: #locust.shape.LoadTestShape
[297]: _modules/locust/shape.html#LoadTestShape.get_current_user_count
[298]: #locust.shape.LoadTestShape.get_current_user_count
[299]: _modules/locust/shape.html#LoadTestShape.get_run_time
[300]: #locust.shape.LoadTestShape.get_run_time
[301]: _modules/locust/shape.html#LoadTestShape.reset_time
[302]: #locust.shape.LoadTestShape.reset_time
[303]: #locust.shape.LoadTestShape.runner
[304]: #locust.runners.Runner
[305]: _modules/locust/shape.html#LoadTestShape.tick
[306]: #locust.shape.LoadTestShape.tick
[307]: _modules/locust/stats.html#RequestStats
[308]: #locust.stats.RequestStats
[309]: _modules/locust/stats.html#RequestStats.get
[310]: #locust.stats.RequestStats.get
[311]: _modules/locust/stats.html#StatsEntry
[312]: #locust.stats.StatsEntry
[313]: _modules/locust/debug.html#run_single_user
[314]: #locust.debug.run_single_user
[315]: #locust.env.Environment
[316]: extending-locust.html#extending-locust
[317]: faq.html
[318]: changelog.html
[319]: https://www.sphinx-doc.org/
[320]: https://github.com/readthedocs/sphinx_rtd_theme
[321]: https://readthedocs.org
