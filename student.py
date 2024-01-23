from enum import Enum
import numpy as np
import pandas as pd
import typing
import scipy.stats as sps
from collections.abc import Callable
from typing import List, Tuple, Dict



precision = 6
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

    def __init__(self, level: int, lang_name: str, lang_id: list):
        self.lvl = level
        self.name = lang_name
        self.id = lang_id


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
    task_row = task_data[(task_data.language_object_id == obj_id) & (task_data.language_session_id == session_id)].sample(1, random_state=generator)
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
        tmp = Language(int(row['course_order']), row['language_name'], [row['language_object_id'], row['language_session_id']])
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

def getLangById(id: list[int]):
    for language in languages:
        if language.id == id:
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
    average_working_time: np.ndarray[np.timedelta64]
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
                 average_session_time: np.ndarray[np.timedelta64], fatige_coef: list[Tuple[LanguageLvl, np.double]],
                 initial_qualification: dict[Language, Qualification], prob_threshold: np.double, 
                 registration_date: np.datetime64, np_seed: int,
                 motivated: bool):
        self.id = ID
        self.state = StudentState.Initial
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


    def __correlation_lvl(self, language: Language) -> np.double:
        lvl = 0
        if self.lang_qualification[language] == Qualification.Middle:
            lvl = LanguageLvl.B2
        if self.lang_qualification[language] == Qualification.High:
            lvl = LanguageLvl.C1

        if language.lvl < self.lang_qualification[language]:
            return 1.0
        if language.lvl == self.lang_qualification[language]:
            return 0.5
        return 0.0
    

    def __get_average_errors_normalized(self, language: Language) -> np.double:
        sum = 0.0
        tasks = 0
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
        for session in self.sessions[getLangById(task.language_id)]:
            if date - session.finish <= np.timedelta64(7, 'd') and task in session.tasks:
                count += 1
            if date - session.finish >= np.timedelta64(3, 'd'):
                existsGlobalDelfa = True
        
        if existsGlobalDelfa and np.around(3.0 / (0.3 * self.__memory_threshold_prob), 0) >= count:
            self.__long_memorized_tasks[task] = date
            return True
        
        return False
        

    def canSolve(self, task: Task) -> bool:
        language = getLangById(task.language_id)
        correlation = self.__get_average_errors_normalized(language)
        initial_score = self.initial_testing[language]
        a_e = self.__get_average_errors_normalized(language)
        inverted_task = task
        inverted_task.solved = not(inverted_task.solved)
        in_short_mem = (task in self.__short_memorized_tasks or 
                        inverted_task in self.__short_memorized_tasks)
        in_long_mem = task in self.__long_memorized_tasks

        probability = 0.01 * correlation + 0.03 * initial_score + 0.1 * self.isLearningRecenlty +\
            0.05 * (1.0 - (len(self.learning_lang) / self.getUniqueLangs())) +\
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
        new_fatigue = self.fatige + 0.01 * learning_language.lvl
        if new_fatigue <= 1.0:
            self.fatige = new_fatigue
        
    def solveSomeTasks(self, limit: np.timedelta64, start_date: np.datetime64) -> np.ndarray[Tuple[Task, np.datetime64]]:
        learning_lang = np.random.Generator.choice(a=self.getUniqueLangs, size=1)
        task_delta = np.timedelta64(np.around(self._generator.normal(loc=self.average_time_per_task * 60) / 60.0, 0), 'm')
        zero = np.timedelta64(0, 's')
        session = Session()
        session.start = start_date
        session.solved = 0
        session.tasks = 0
        result = []
        while (limit - task_delta > zero) and (self.fatige < 1.0):
            limit -= task_delta
            start_date += task_delta
            task = getTask(learning_lang, self._generator)
            resultForTask = self.canSolve(task)
            logResult(task, start_date - task_delta, start_date)
            task.solved = resultForTask
            result.append((task, start_date))
            self.increaseFatige(learning_lang)
            task_delta = np.timedelta64(np.around(self._generator.normal(loc=self.average_time_per_task * 60) / 60.0, 0), 'm')
            
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
                    start_date - task_data - self.__short_memorized_tasks[task] > np.timedelta64(1, 'h'):
                    del self.__short_memorized_tasks[task]
                    self.tryToAddToShortMemory(task, start_date)

                if inverted_task in self.__short_memorized_tasks and \
                    start_date - task_data - self.__short_memorized_tasks[task] > np.timedelta64(1, 'h'):
                    del self.__short_memorized_tasks[inverted_task]
                    self.tryToAddToShortMemory(task, start_date)
            
            if not(task.solved and (task in self.__long_memorized_tasks)):
                self.tryToAddToLongMemory(task, start_date)
            else:
                if task in self.__long_memorized_tasks and \
                    start_date - task_data - self.__long_memorized_tasks[task] > np.timedelta64(1, 'y'):
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

        session.language = getLangById(result[-1][0].language_id)

        self.sessions[session.language].append(session)

        self.fatige = 0.0
        return np.array(result)
    
    

    def calcWorkDuration(self, delta: np.timedelta64) -> np.timedelta64:
        minimum = 5.0 * 60.0
        result = 0.0
        while (result <= minimum) or (result > delta):
            result = self._generator.normal(loc = self.average_working_time * 60)
        return np.timedelta64(np.around(result, 0), 's')
        

    def increasePositiveProb(self, ):
        self.isLearningRecenlty = True

    def decreasePositiveProb(self, ):
        self.isLearningRecenlty = False
    
    def call(self, ) -> np.datetime64:
        # initial state processing
        if self.state == StudentState.Initial:
            self.state = calcNextState([0.6, 0.3, 0.08, 0.02], self._generator)
            if self.state == StudentState.Inactive:
                self._next_event_time = self._generator.uniform(low = self._next_event_time, high = self._next_event_time \
                                                          + np.timedelta64(100, 'D'))
            elif self.state == StudentState.Dead:
                self._next_event_time = np.datetime64('1970-01-01T00:00')

            elif self.state == StudentState.Learning:
                self._next_event_time = self._generator.uniform(low = self._next_event_time, high = self._next_event_time \
                                                          + np.timedelta64(10, 'D'))

                self.increasePositiveProb()
            
            elif self.state == StudentState.Working:
                self.setWorkingDates(self._next_event_time)
                self._next_event_time = self.__activity_tmstmp[0]
                self.__activity_tmstmp_count = 0
        
        # current state is working
        elif self.state == StudentState.Working:
            working_duration = self.calcWorkDuration(
                self.__activity_tmstmp[self.__activity_tmstmp + 1] - self.__next_event_time)
            self.solveSomeTasks(working_duration, self._next_event_time)
            
            self._next_event_time += working_duration
            prob_i = np.array([
                0.0,                             #Working 
                np.around(1.0 / 3.0, precision), #Learning
                np.around(1.0 / 3.0, precision), #Inactive
                np.around(1.0 / 3.0, precision), #Dead
            ])
            langs = self.getUniqueLangs()
            result_prob = np.array([0.0, 0.0, 0.0, 0.0])         
            vectors = np.array([prob_i for _ in len(langs)])
            for i, language in enumerate(langs):
                if language.lvl < LanguageLvl.B1:
                    vectors[i][2] += 0.1
                    vectors[i][1] -= 0.1
                if self.motivated:
                    vectors[i][1] += 0.2
                    vectors[i][2] -= 0.1
                    vectors[i][3] -= 0.1
                if self.__activity_tmstmp[self.__activity_tmstmp_count + 1] \
                - self.__activity_tmstmp[self.__activity_tmstmp_count] <= np.timedelta64(24, 'H'):
                    vectors[i][1] -= 0.05
                    vectors[i][2] += 0.05
                if self.__activity_tmstmp[self.__activity_tmstmp_count + 1] \
                - self.__activity_tmstmp[self.__activity_tmstmp_count] > np.timedelta64(24, 'H'):
                    vectors[i][1] += 0.03
                    vectors[i][2] -= 0.03
                for j in len(vectors[i]):
                    result_prob[j] += vectors[i][j]

            for i in range(len(result_prob)):
                result_prob[i] = result_prob[i] / np.double(len(langs))

            self.state = calcNextState(result_prob, self._generator)
            if self.state == StudentState.Dead:
                self._next_event_time = np.datetime64('1970-01-01T00:00')
            
            self.__activity_tmstmp_count += 1
        
        # other states
        elif self.state != StudentState.Dead:
            if self.state == StudentState.Learning:
                self.increasePositiveProb()
            elif self.state == StudentState.Inactive:
                self.decreasePositiveProb()
            self.state = calcNextState([1.0, 0.0, 0.0, 0.0], self._generator)
            if self.state == StudentState.Working:
                self._next_event_time = self.__activity_tmstmp[self.__activity_tmstmp_count]
        
        return self._next_event_time
    


def main():
    print(f"{languages[0].name} {languages[0].lvl} {languages[0].id}")
    first_student = Student(
        ID=1, 
        learning_langs=[languages[0]], 
        native_lang=Language(level=5, lang_name='русский', lang_id=[12, 0]),
        average_task_time=np.array([(languages[0], 40)]),
        initial_tests=np.array([(languages[0], 0.3)]),
        average_session_time=np.array([(languages[0], 15 * 60)]),
        fatige_coef=[
            (j, 0.01 * j) for j in range(1, 7)
        ],
        initial_qualification={
            languages[0]: Qualification.Low
        },
        prob_threshold=0.4,
        registration_date=np.datetime64('2024-01-05'),
        np_seed=101010101,
        motivated=True
    )
    for i in range(10):
        print(i)
        first_student.call()
main()