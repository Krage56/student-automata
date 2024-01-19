from enum import Enum
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


import numpy as np
def calcNextState(vec) -> StudentState:
    random_num = np.random.uniform(size=1)[0]
    bottom_border = 0.0
    for i in range(len(vec)):
        if bottom_border <= random_num <= sum(vec[:i + 1]):
            return i + 1
        else:
            bottom_border += vec[i]




import pandas as pd
task_data = pd.read_csv('./all_real_task_data.csv')
def getTask(learning_lang: Language) -> Task:
    obj_id = learning_lang.id[0]
    session_id = learning_lang.id[1]
    task_row = task_data[(task_data.language_object_id == obj_id) & (task_data.language_session_id == session_id)].sample(1)
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

import numpy as np
import typing
import scipy.stats as sps
from collections.abc import Callable

class Student:
    id: np.int
    state: StudentState
    fatige: np.double
    fatige_per_task: list[tuple(LanguageLvl, np.double)]
    learning_langs: list[Language]
    initial_testing: set[Language]
    native_language: Language
    average_time_per_task: np.array[tuple(Language, np.double)]
    average_working_time: np.array[np.timedelta64]
    lang_qualification: dict[Language, Qualification]
    activity_times: list[tuple(Topic, np.datetime64)]
    motivated: bool
    sessions: list[tuple(np.datetime64, np.datetime64)]
    active_dates_generator: Callable[[np.datetime64], list[np.datetime64]]
    isLearningRecenlty: bool
    __memory_threshold_prob: np.double
    __long_memorized_topics: dict[Topic, np.datetime64]
    __short_memorized_topic: dict[Topic, np.datetime64]
    _seed: np.int
    _next_event_time: np.datetime64
    __activity_tmstmp: np.array[np.datetime64]
    __activity_tmstmp_count: np.int

    def __init__(self, ID: int, learning_langs: list[Language], native_lang: Language, 
                average_task_time: np.array[tuple(Language, np.double)], initial_tests: set[Language],
                 average_session_time: np.array[np.datetime64], fatige_coef: list[tuple(LanguageLvl, np.double)],
                 initial_qualification: dict[Language, Qualification], prob_threshold: np.double, 
                 registration_date: np.datetime64, 
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
        
        self.__long_memorized_topics = dict()
        self.__short_memorized_topics = dict() 
        
    # def canSolve(task: Task):
    #     # TODO
        
    
    def getUniqueLangs() -> np.array[Language]:
        lang_dict = dict()
        lang_names = set()
        for language in typing.Self.learning_langs:
            if language.name in lang_names:
                if language.lvl > lang_dict[language.name]:
                    lang_dict[language.name] = language
            else:
                lang_dict.update([language.name, language])
                lang_names.add(language.name)
        return np.array(lang_dict.values())
    
    def setWorkingDates(start_date: np.datetime64) -> np.array[np.datetime64]:
        typing.Self.__activity_tmstmp = typing.Self.active_dates_generator(start_date)
        typing.Self._activity_tmstmp_count = 0
        return typing.Self.__activity_tmstmp
    

    def increaseFatige(learning_language: Language):
        new_fatigue = typing.Self.fatige + 0.01 * learning_language.lvl
        if new_fatigue <= 1.0:
            typing.Self.fatige = new_fatigue
        
    def solveSomeTasks(limit: np.timedelta64, start_date: np.datetime64) -> np.array[tuple(Task, np.datetime64)]:
        learning_lang = np.random.Generator.choice(a=typing.Self.getUniqueLangs, size=1)
        task_delta = np.timedelta64(np.around(np.random.normal(loc=typing.Self.average_time_per_task * 60) / 60.0, 0), 'm')
        zero = np.timedelta64(0, 's')
        solved = 0
        result = []
        while (limit - task_delta > zero) and (typing.Self.fatige < 1.0):
            limit -= task_delta
            start_date += task_delta
            task = getTask(learning_lang)
            resultForTask = typing.Self.canSolve(task)
            logResult()
            task.solved = resultForTask
            result.append((task, start_date))
            typing.Self.increaseFatige(learning_lang)
            task_delta = np.timedelta64(np.around(np.random.normal(loc=typing.Self.average_time_per_task * 60) / 60.0, 0), 'm')
            solved += 1

            
        if solved == 0:
            task_delta = limit
            task = getTask(learning_lang)
            resultForTask = typing.Self.canSolve()
            logResult()
            task.solved = resultForTask
            result.append((task, start_date))
        
        typing.Self.fatige = 0.0
        return np.array(result)
    
    

    def calcWorkDuration(delta: np.timedelta64) -> np.timedelta64:
        minimum = 5.0 * 60.0
        result = 0.0
        while (result <= minimum) or (result > delta):
            result = np.random.normal(loc = typing.Self.average_working_time * 60)
        return np.timedelta64(np.around(result, 0), 's')
        

    def increasePositiveProb():
        typing.Self.isLearningRecenlty = True

    def decreasePositiveProb():
        typing.Self.isLearningRecenlty = False
    
    def call() -> np.datetime64:
        # initial state processing
        if typing.Self.state == StudentState.Initial:
            typing.Self.state = calcNextState([0.6, 0.3, 0.08, 0.02])
            if typing.Self.state == StudentState.Inactive:
                typing.Self._next_event_time = np.random.uniform(low = typing.Self._next_event_time, high = typing.Self._next_event_time \
                                                          + np.timedelta64(100, 'D'))
            elif typing.Self.state == StudentState.Dead:
                typing.Self._next_event_time = np.datetime64('1970-01-01T00:00')

            elif typing.Self.state == StudentState.Learning:
                typing.Self._next_event_time = np.random.uniform(low = typing.Self._next_event_time, high = typing.Self._next_event_time \
                                                          + np.timedelta64(10, 'D'))

                typing.Self.increasePositiveProb()
            
            elif typing.Self.state == StudentState.Working:
                typing.Self.setWorkingDates(typing.Self._next_event_time)
                typing.Self._next_event_time = typing.Self.__activity_tmstmp[0]
                typing.Self.__activity_tmstmp_count = 0
        
        # current state is working
        elif typing.Self.state == StudentState.Working:
            working_duration = typing.Self.calcWorkDuration(
                typing.Self.__activity_tmstmp[typing.Self.__activity_tmstmp + 1] - typing.Self.__next_event_time)
            typing.Self.solveSomeTasks(working_duration, typing.Self._next_event_time)
            typing.Self._next_event_time += working_duration
            prob_i = np.array([
                0.0,                             #Working 
                np.around(1.0 / 3.0, precision), #Learning
                np.around(1.0 / 3.0, precision), #Inactive
                np.around(1.0 / 3.0, precision), #Dead
            ])
            langs = typing.Self.getUniqueLangs()
            result_prob = np.array([0.0, 0.0, 0.0, 0.0])         
            vectors = np.array([prob_i for _ in len(langs)])
            for i, language in enumerate(langs):
                if language.lvl < LanguageLvl.B1:
                    vectors[i][2] += 0.1
                    vectors[i][1] -= 0.1
                if typing.Self.motivated:
                    vectors[i][1] += 0.2
                    vectors[i][2] -= 0.1
                    vectors[i][3] -= 0.1
                if typing.Self.__activity_tmstmp[typing.Self.__activity_tmstmp_count + 1] \
                - typing.Self.__activity_tmstmp[typing.Self.__activity_tmstmp_count] <= np.timedelta64(24, 'H'):
                    vectors[i][1] -= 0.05
                    vectors[i][2] += 0.05
                if typing.Self.__activity_tmstmp[typing.Self.__activity_tmstmp_count + 1] \
                - typing.Self.__activity_tmstmp[typing.Self.__activity_tmstmp_count] > np.timedelta64(24, 'H'):
                    vectors[i][1] += 0.03
                    vectors[i][2] -= 0.03
                for j in len(vectors[i]):
                    result_prob[j] += vectors[i][j]

            for i in range(len(result_prob)):
                result_prob[i] = result_prob[i] / np.double(len(langs))

            typing.Self.state = calcNextState(result_prob)
            if typing.Self.state == StudentState.Dead:
                typing.Self._next_event_time = np.datetime64('1970-01-01T00:00')
            
            typing.Self.__activity_tmstmp_count += 1
        
        # other states
        elif typing.Self.state != StudentState.Dead:
            if typing.Self.state == StudentState.Learning:
                typing.Self.increasePositiveProb()
            elif typing.Self.state == StudentState.Inactive:
                typing.Self.decreasePositiveProb()
            typing.Self.state = calcNextState([1.0, 0.0, 0.0, 0.0])
            if typing.Self.state == StudentState.Working:
                typing.Self._next_event_time = typing.Self.__activity_tmstmp[typing.Self.__activity_tmstmp_count]
        
        return typing.Self._next_event_time