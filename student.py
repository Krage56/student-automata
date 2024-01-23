from enum import Enum
import numpy as np
from datetime import datetime
import pandas as pd
import scipy.stats as sps
from collections.abc import Callable
from typing import List, Tuple, Dict

precision = 6

class StudentState(Enum):
    Initial = 0
    Working = 1
    Learning = 2
    Inactive = 3
    Dead = 4

class Qualification(Enum):
    Low = 0
    Middle = 1
    High = 2

class TaskType(Enum):
    closureInput = 0
    closureSelect = 1
    translate = 2



class Topic:
    id: list[int]
    name: str


class Task:
    type: TaskType
    id: list[int]
    course_id: list[list[int]]
    topic_id: list[list[int]]
    language_id: list[int]
    solved: bool


class LanguageLvl(Enum):
    A1 = 1
    A2 = 2
    B1 = 3
    B2 = 4
    C1 = 5
    C2 = 6

class Language:
    lvl: LanguageLvl
    name: str
    id: list[int]
    course: list[int]

    def __init__(self, level: int, lang_name: str, lang_id: list, course_id: list):
        self.lvl = level
        self.name = lang_name
        self.id = lang_id
        self.course = course_id



class Session:
    start: np.datetime64
    finish: np.datetime64
    language: Language
    tasks: int
    solved: int
    solved_tasks: list[Task]


def calcNextState(vec, generator) -> StudentState:
    random_num = generator.uniform(size=1)[0]
    bottom_border = 0.0
    for i in range(len(vec)):
        if bottom_border <= random_num <= sum(vec[:i + 1]):
            return i + 1
        else:
            bottom_border += vec[i]




task_data = pd.read_csv('./all_real_task_data.csv')
def getTask(learning_lang: Language, generator: np.random.Generator) -> Task:
    obj_id = learning_lang.id[0]
    session_id = learning_lang.id[1]
    course_order = learning_lang.lvl.value
    task_row = task_data[(task_data.language_object_id == obj_id) &\
                          (task_data.language_session_id == session_id) &\
                              (task_data.course_order == course_order)]\
                                .sample(1, random_state=generator)
    result = Task()
    result.id = [task_row.iloc[0]['task_object_id'], task_row.iloc[0]['task_session_id']]
    result.course_id = [task_row.iloc[0]['course_object_id'], task_row.iloc[0]['course_session_id']]
    result.topic_id = [task_row.iloc[0]['topic_object_id'], task_row.iloc[0]['topic_session_id']]
    result.language_id = [task_row.iloc[0]['language_object_id'], task_row.iloc[0]['language_session_id']]
    type = task_row.iloc[0]['task_type']
    if type == 'completeSentence':
        result.type = TaskType.closureSelect
    else:
        result.type = TaskType.closureInput
    result.solved = False
    return result

def languageFactory(df: pd.DataFrame)->list[Language]:
    result = []
    for _, row in df.iterrows():
        tmp = Language(
            LanguageLvl(int(row['course_order'])), 
            row['language_name'], 
            [
                row['language_object_id'], 
                row['language_session_id']
            ],
            [
                row['course_object_id'],
                row['course_session_id']
            ]
        )
        result.append(tmp)
    return result


languages = languageFactory(
    task_data.drop_duplicates(
        subset=[
            'language_object_id', 
            'language_session_id', 
            'course_order', 
            'language_name',
        ]
    )
)

def getLangById(id: list[int], course: list[int]):
    for language in languages:
        if language.id == id and language.course == course:
            return language



def logResult(task: Task, start: np.datetime64, finish: np.datetime64):
    print(f"id={task.id}, course_id={task.course_id}, solved={task.solved}, start={start}, finish={finish}")

class Student:
    id: int
    state: StudentState
    fatige: np.double
    fatige_per_task: list[Tuple[LanguageLvl, np.double]]
    learning_langs: list[Language]
    initial_testing: dict[Language, np.double]
    native_language: Language
    average_time_per_task: np.ndarray[Tuple[Language, np.double]]
    average_working_time: np.timedelta64
    lang_qualification: dict[Language, Qualification]
    activity_times: list[Tuple[Topic, np.datetime64]]
    motivated: bool
    sessions: dict[Language, list[Session]]
    active_dates_generator: Callable[[np.datetime64], list[np.datetime64]]
    isLearningRecenlty: bool
    __memory_threshold_prob: np.double
    __long_memorized_tasks: dict[Task, np.datetime64]
    __short_memorized_tasks: dict[Task, np.datetime64]
    _seed: int
    _generator: np.random.Generator
    _next_event_time: np.datetime64
    __activity_tmstmp: np.ndarray[np.datetime64]
    __activity_tmstmp_count: int

    def __init__(self, ID: int, learning_langs: list[Language], native_lang: Language, 
                average_task_time: np.ndarray[Tuple[Language, np.double]], initial_tests: dict[Language, np.double],
                 average_session_time: np.timedelta64, fatige_coef: list[Tuple[LanguageLvl, np.double]],
                 initial_qualification: dict[Language, Qualification], prob_threshold: np.double, 
                 registration_date: np.datetime64, np_seed: int,
                 motivated: bool):
        self.id = ID
        self.state = StudentState.Initial.value
        self.fatige = 0.0
        self.fatige_per_task = fatige_coef
        self.native_language = native_lang
        self.initial_testing = initial_tests
        self.learning_langs = learning_langs
        self.average_time_per_task = average_task_time
        self.lang_qualification = initial_qualification
        self.__memory_threshold_prob = prob_threshold
        self._next_event_time = registration_date
        self.motivated = motivated
        self.average_working_time = average_session_time
        self.isLearningRecenlty = False

        self.sessions = {}
        for language in self.getUniqueLangs():
            self.sessions.update({language: []})
        
        self._seed = np_seed
        self._generator = np.random.default_rng(self._seed)
        
        self.__long_memorized_tasks = dict()
        self.__short_memorized_tasks = dict()

        self.__activity_tmstmp = np.array([])
        self.__activity_tmstmp_count = 0


    def __correlation_lvl(self, language: Language) -> np.double:
        lvl = 0
        if self.lang_qualification[language].value == Qualification.Middle.value:
            lvl = LanguageLvl.B2.value
        if self.lang_qualification[language].value == Qualification.High.value:
            lvl = LanguageLvl.C1.value

        if language.lvl.value < self.lang_qualification[language].value:
            return 1.0
        if language.lvl.value == self.lang_qualification[language].value:
            return 0.5
        return 0.0
    

    def __get_average_errors_normalized(self, language: Language) -> np.double:
        sum = 0.0
        tasks = 0
        if len(self.sessions[language]) == 0:
            return 1.0
        for session in self.sessions[language]:
            sum += (session.solved / session.tasks)
            tasks += session.tasks
        return np.around(sum / tasks, precision)

    def tryToAddToShortMemory(self, task: Task, date: np.datetime64):
        probability = self.__memory_threshold_prob
        if not(task.solved):
            probability *= 0.5
        point = self._generator.uniform(size=1)[0]
        if point <= probability:
            self.__long_memorized_tasks[task] = date
            return True
        return False
    
    def tryToAddToLongMemory(self, task: Task, date: np.datetime64) -> bool:
        count = 0
        existsGlobalDelfa = False
        for session in self.sessions[getLangById(task.language_id, task.course_id)]:
            if date - session.finish <= np.timedelta64(7, 'D') and task in session.solved_tasks:
                count += 1
            if date - session.finish >= np.timedelta64(3, 'D'):
                existsGlobalDelfa = True
        
        if existsGlobalDelfa and np.around(3.0 / (0.3 * self.__memory_threshold_prob), 0) >= count:
            self.__long_memorized_tasks[task] = date
            return True
        
        return False
        

    def canSolve(self, task: Task) -> bool:
        language = getLangById(task.language_id, task.course_id)
        correlation = self.__correlation_lvl(language)
        initial_score = self.initial_testing[language]
        a_e = self.__get_average_errors_normalized(language)
        inverted_task = task
        inverted_task.solved = not(inverted_task.solved)
        in_short_mem = (task in self.__short_memorized_tasks or 
                        inverted_task in self.__short_memorized_tasks)
        in_long_mem = task in self.__long_memorized_tasks

        probability = 0.01 * correlation + 0.03 * initial_score + 0.1 * self.isLearningRecenlty +\
            0.05 * (1.0 - (len(self.learning_langs) / len(self.getUniqueLangs()))) +\
            0.2 * (1.0 - self.fatige) + 0.02 * (1 - a_e) + 0.3 * in_short_mem + 0.2 * in_long_mem
        
        return probability <= self._generator.uniform(size=1)[0]
    
    def getUniqueLangs(self, ) -> np.ndarray[Language]:
        lang_dict = dict()
        lang_names = set()
        for language in self.learning_langs:
            if language.name in lang_names:
                if language.lvl > lang_dict[language.name]:
                    lang_dict[language.name] = language
            else:
                lang_dict.update({language.name: language})
                lang_names.add(language.name)
        return np.array([el for el in lang_dict.values()])
    
    def setWorkingDates(self, start_date: np.datetime64) -> np.ndarray[np.datetime64]:
        self.__activity_tmstmp = self.active_dates_generator(start_date)
        self._activity_tmstmp_count = 0
        return self.__activity_tmstmp
    

    def increaseFatige(self, learning_language: Language):
        new_fatigue = self.fatige + 0.01 * learning_language.lvl.value
        if new_fatigue <= 1.0:
            self.fatige = new_fatigue
        
    def solveSomeTasks(self, limit: np.timedelta64, start_date: np.datetime64) -> np.ndarray[Tuple[Task, np.datetime64]]:
        learning_lang = self._generator.choice(a=self.getUniqueLangs(), size=1)[0]
        average_time = 0.0
        for lang, time in self.average_time_per_task:
            if lang == learning_lang:
                average_time = time
                break
        
        task_delta = np.timedelta64(int(np.around(self._generator.normal(loc=average_time), 0)), 's')
        zero = np.timedelta64(0, 'm')
        session = Session()
        session.start = start_date
        session.solved = 0
        session.tasks = 0
        session.solved_tasks = []
        result = []
        while (limit - task_delta > zero) and (self.fatige < 1.0):
            limit -= task_delta
            start_date += task_delta
            task = getTask(learning_lang, self._generator)
            resultForTask = self.canSolve(task)
            task.solved = resultForTask
            logResult(task, start_date - task_delta, start_date)
            result.append((task, start_date))
            self.increaseFatige(learning_lang)
            task_delta = np.timedelta64(int(np.around(self._generator.normal(loc=average_time), 0)), 's')
            
            session.tasks += 1
            if task.solved:
                session.solved += 1
                session.solved_tasks.append(task)

            inverted_task = task
            inverted_task.solved = not(task.solved)
            if not(task in self.__short_memorized_tasks or inverted_task in self.__short_memorized_tasks):
                self.tryToAddToShortMemory(task, start_date)
            else:
                if task in self.__short_memorized_tasks and \
                    start_date - task_delta - self.__short_memorized_tasks[task] > np.timedelta64(1, 'h'):
                    del self.__short_memorized_tasks[task]
                    self.tryToAddToShortMemory(task, start_date)

                if inverted_task in self.__short_memorized_tasks and \
                    start_date - task_delta - self.__short_memorized_tasks[task] > np.timedelta64(1, 'h'):
                    del self.__short_memorized_tasks[inverted_task]
                    self.tryToAddToShortMemory(task, start_date)
            
            if not(task.solved and (task in self.__long_memorized_tasks)):
                self.tryToAddToLongMemory(task, start_date)
            else:
                if task in self.__long_memorized_tasks and \
                    start_date - task_delta - self.__long_memorized_tasks[task] > np.timedelta64(365*24*60, 's'):
                    del self.__long_memorized_tasks[task]
                    self.tryToAddToLongMemory(task, start_date)

            
        if session.tasks == 0:
            task_delta = limit
            task = getTask(learning_lang, self._generator)
            resultForTask = self.canSolve()
            logResult(task, start_date - task_delta, start_date)
            task.solved = resultForTask
            result.append((task, start_date))
            
            session.tasks += 1
            if task.solved:
                session.solved += 1
            
            inverted_task = task
            inverted_task.solved = not(task.solved)
            if not(task in self.__short_memorized_tasks or inverted_task in self.__short_memorized_tasks):
                self.tryToAddToShortMemory(task, start_date)
            if not(task.solved and (task in self.__long_memorized_tasks)):
                self.tryToAddToLongMemory(task, start_date)

            session.finish = start_date + limit
        
        else:
            session.finish = start_date

        session.language = getLangById(result[-1][0].language_id, result[-1][0].course_id)

        self.sessions[session.language].append(session)

        self.fatige = 0.0
        return np.array(result)
    
    

    def calcWorkDuration(self, delta: np.timedelta64) -> np.timedelta64:
        minimum = 5.0 * 60.0
        result = 0.0
        double_delta = delta.astype('double')
        while (result <= minimum) or (result > double_delta):
            result = self._generator.normal(loc = self.average_working_time.astype('double'))
        return np.timedelta64(int(np.around(result, 0)), 's')
        

    def increasePositiveProb(self, ):
        self.isLearningRecenlty = True

    def decreasePositiveProb(self, ):
        self.isLearningRecenlty = False
    
    def call(self, ) -> np.datetime64:
        # initial state processing
        if self.state == StudentState.Initial.value:
            self.state = calcNextState([0.6, 0.3, 0.08, 0.02], self._generator)
            if self.state == StudentState.Inactive.value:
                self._next_event_time = np.array([self._generator.uniform(low = self._next_event_time.astype('double'), 
                                          high = (self._next_event_time + np.timedelta64(100, 'D')).astype('double'))]).astype('datetime64[m]')[0]
            elif self.state == StudentState.Dead.value:
                self._next_event_time = np.datetime64('1970-01-01T00:00')

            elif self.state == StudentState.Learning.value:
                self._next_event_time = np.array([self._generator.uniform(
                    low = self._next_event_time.astype('double'), 
                    high = (self._next_event_time + np.timedelta64(10, 'D')).astype('double')
                )]).astype('datetime64[m]')[0]

                self.increasePositiveProb()
            
            elif self.state == StudentState.Working.value:
                self.setWorkingDates(self._next_event_time)
                self._next_event_time = self.__activity_tmstmp[0]
                self.__activity_tmstmp_count = 0
        
        # current state is working
        elif self.state == StudentState.Working.value:
            working_duration = self.calcWorkDuration(
                self.__activity_tmstmp[self.__activity_tmstmp_count + 1] - self._next_event_time)
            self.solveSomeTasks(working_duration, self._next_event_time)
            
            self._next_event_time += working_duration
            prob_i = np.array([
                0.29, #Working 
                0.29, #Learning
                0.29, #Inactive
                0.13, #Dead
            ])
            langs = self.getUniqueLangs()
            result_prob = np.array([0.0, 0.0, 0.0, 0.0])         
            vectors = np.array([prob_i for _ in langs])
            for i, language in enumerate(langs):
                if language.lvl.value < LanguageLvl.B1.value:
                    vectors[i][2] += 0.1
                    vectors[i][1] -= 0.1
                if self.motivated:
                    vectors[i][1] += 0.2
                    vectors[i][2] -= 0.1
                    vectors[i][3] -= 0.1
                if self.__activity_tmstmp[self.__activity_tmstmp_count + 1] \
                - self.__activity_tmstmp[self.__activity_tmstmp_count] <= np.timedelta64(24, 'h'):
                    vectors[i][1] -= 0.05
                    vectors[i][2] += 0.05
                if self.__activity_tmstmp[self.__activity_tmstmp_count + 1] \
                - self.__activity_tmstmp[self.__activity_tmstmp_count] > np.timedelta64(24, 'h'):
                    vectors[i][1] += 0.03
                    vectors[i][2] -= 0.03
                for j in range(len(vectors[i])):
                    result_prob[j] += vectors[i][j]

            for i in range(len(result_prob)):
                result_prob[i] = result_prob[i] / np.double(len(langs))

            self.state = calcNextState(result_prob, self._generator)
            if self.state == StudentState.Dead.value:
                self._next_event_time = np.datetime64('1970-01-01T00:00')
            
            self.__activity_tmstmp_count += 1
        
        # other states
        elif self.state != StudentState.Dead.value:
            if self.state == StudentState.Learning.value:
                self.increasePositiveProb()
            elif self.state == StudentState.Inactive.value:
                self.decreasePositiveProb()
            self.state = calcNextState([1.0, 0.0, 0.0, 0.0], self._generator)
            if self.state == StudentState.Working.value:
                if len(self.__activity_tmstmp) == 0:
                    self.setWorkingDates(self._next_event_time)
                
                self._next_event_time = self.__activity_tmstmp[self.__activity_tmstmp_count]
        
        return self._next_event_time
    


def dates_range(start, end, date_format, distribution_size, scale_ratio, seed):
    # Converting to timestamp
    start = datetime.strptime(start, date_format).timestamp()
    end = datetime.strptime(end, date_format).timestamp()
    
    # Generate Normal Distribution
    mu = (end - start) / 2 + start
    sigma = (end - start) * scale_ratio
    generator = np.random.default_rng(seed)
    a_transformed, b_transformed = (start - mu) / sigma, (end - mu) / sigma
    total_distribution = sps.truncnorm(
        loc=mu, 
        scale=sigma, 
        a=a_transformed,
        b=b_transformed
    ).rvs(random_state=generator, size=distribution_size,)

    # Sort and Convert back to datetime
    sorted_distribution = np.sort(total_distribution)
    date_range = [np.datetime64(datetime.fromtimestamp(t)) for t in sorted_distribution]
    # print(date_range)
    for date in date_range:
        print(date)
    return date_range


def main():
    print(f"{languages[0].name} {languages[0].lvl} {languages[0].id}")
    first_student = Student(
        ID=1, 
        learning_langs=[languages[2]], 
        native_lang=Language(level=LanguageLvl.A1, lang_name='русский', lang_id=[12, 0], course_id=None),
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
    first_student.active_dates_generator = lambda x: dates_range(
        start=np.datetime_as_string(x), 
        end=np.datetime_as_string(x + np.timedelta64(1 * 365 * 24 * 60, 'm')),
        distribution_size=365*2,
        scale_ratio=0.02,
        seed=556161691000000000000522,
        date_format = '%Y-%m-%dT%H:%M'
    )
    for i in range(100):
        print(i)
        print(first_student.call())
main()