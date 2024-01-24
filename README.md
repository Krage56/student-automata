Git repo for simple imitator of student on languages learning platform. At first sight it's a simple probability automata with scheduler that controls sequence of the Students' objects actions.

There are a few probability distributions that dictate the rules of students' working times: exponential, normal and some else in the future.

Other details will be provided with the upgrades of the notebook.

The code is a part of coursework on the magister program 'System programming' in the HSE University, Faculty of Computer Science.


# Student class description
Student is the main class of the imitator. Example of initialization:
```python
first_student = Student(
        ID=1, 
        learning_langs=[languages[2]], 
        native_lang=Language(
            level=LanguageLvl.A1, 
            lang_name='русский', 
            lang_id=[12, 0], 
            course_id=None
        ),
        average_task_time=np.array([(languages[2], 40)]),
        initial_tests={languages[2]: 0.3},
        average_session_time=np.timedelta64(15 * 60, 's'),
        fatige_coef=[
            (j, 0.01 * j) for j in range(1, 7)
        ],
        initial_qualification={
            languages[2]: Qualification.Middle
        },
        prob_threshold=0.3,
        registration_date=np.datetime64('2024-01-05T15:30'),
        np_seed=5649849415165165160000000000000000000000000556162516216251915165,
        motivated=True
    )
```
where `ID` - identificator of the student, `learning_langs` - python list of the learning languages (represented by the `Language` class), `native_lang` - native language of the student (now is useless), `average_task_time` - list of tupled pairs with `Language` instances and average *time in seconds* that are required for task, `initial_tests` - dict with results of the initial test for every language learning by the `student`, `average_session_time` - average *time in seconds* for learning session for every language, `fatige_coef` - list of the numbers which describes increaces of student's exhaustion level on the every learning language with every solved (right or wrong) task, `prob_threshold` - characteristic laying in $(0, 1)$ describes the power of memorization of the student, `registration_date` - date of the student's registration in the language learning system (may not be equal for the first 'working' day), `np_seed` - seed for the random generator is used to pin the random distributions, `motivated` - feature of the model that represents student's motivation.

To use the `Student` object after initialization you must set up a function that will generates probable dates of the `events`. The example above:
```python
first_student.active_dates_generator = lambda x: dates_range(
        start=np.datetime_as_string(x), 
        end=np.datetime_as_string(x + np.timedelta64(1 * 365 * 24 * 60, 'm')),
        distribution_size=365*2,
        scale_ratio=0.02,
        seed=556161691000000000000522,
        date_format = '%Y-%m-%dT%H:%M'
    )
```
*you must use the same `date_format` and timedelta in minutes* for custom `active_dates_generator`.

## Methods
- `call()` - the most important method that is called when you would to execute some `event` and get the time of the next one (that is returned by the method);
- the rest of methods are not supposed to be called by the user manually.

## Model description
The model has a number of assumptions:
1. Student learns the only one language from the `learning_langs` per session and the language is choosen according to the uniform distribution;
1. Student has only 4 states:
    - Working: student is solving tasks;
    - Learning: student have gone away from the system and with the next event is returning with higher probability of the right solving for tasks;
    - Inactive: student have gone till next `event`;
    - Dead: student left the system.
1. Tasks are not correlated with each other by topics and other features;
1. Student can memorize only tasks, not the topic or something else;
1. Motivated student have more events then unmotivated;
1. Student can't change his qualification;
1. If there is a plenty of courses for one languages in student's model then user is solving the course with the highest order;


## Probability for task solving
$0.01 \cdot correlation + 0.03 \cdot initialScore + 0.1 \cdot isLearningRecently + 0.05 \cdot \left(1 - \frac{\left|uniqueLearningLanguages\right|}{\left|learningLanguages\right|}\right)  + 0.2 \cdot \left(1 - fatige\right) + 0.02 \cdot (1 - averageErrors) + 0.3 \cdot inShortMemory + 0.2 \cdot inLongMemory$,

where `correlation` is correlation between defined course level and student's level in the $[0;1]$ (another words if student's qualification is higher then the course level then correlation should be higher then zero); `isLearningRecently` is the boolean feature that displaying that previous state was 'Learning' and now probability of right solving is higher; `learningLanguage` is a set of all language courses meanwhile `uniqueLearningLanguages` is a set of languages that student is studying; `averageErrors` is the number in $[0,1]$ represents the average proportion of errors per session; `inShortMemory` is a boolean feature that is True when the task in the short student's memory; `inLongMemory` is similar with `inShortMemory`.

## Calculation of the state
TODO