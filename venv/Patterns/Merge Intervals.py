import sys
from collections import defaultdict
import heapq


# Given a list of intervals, merge all the overlapping
# intervals to produce a list that has only mutually exclusive intervals.
class Interval:
    def __init__(self, start=0, end=0):
        self.start = start
        self.end = end


def mergeIntervals(arr):
    output = []
    if len(arr) < 2:
        for i in arr:
            print(i[0], i[1])
        return
    intervals = []
    for i in arr:
        interval = Interval(i[0], i[1])
        intervals.append(interval)
    intervals = sorted(intervals, key=lambda x: x.start)
    start = intervals[0].start
    end = intervals[0].end
    for i in intervals[1:]:
        if i.start <= end:
            end = max(end, i.end)
        else:
            output.append(Interval(start, end))
            start = i.start
            end = i.end
    output.append(Interval(start, end))
    for i in output:
        print(i.start, i.end)


# arr=[[1,3],[4,6], [5,7], [8,12]]
# mergeIntervals(arr)


def insertIntervals(arr, interval):
    output = []
    if len(arr) < 2:
        for i in arr:
            print(i[0], i[1])
        return
    intervals = []
    for i in arr:
        inter = Interval(i[0], i[1])
        intervals.append(inter)
    start = intervals[0].start
    end = intervals[0].end
    for i in intervals[1:]:
        if interval[0] <= end:
            start = min(start, interval[0])
            end = max(end, interval[1])
        if i.start <= end:
            end = max(end, i.end)
        else:
            output.append(Interval(start, end))
            start = i.start
            end = i.end
    output.append(Interval(start, end))
    for i in output:
        print(i.start, i.end)


# Intervals=[[2,3],[5,7]]
# interval=[1,4]
# insertIntervals(Intervals,interval)

# Given two lists of intervals, find the intersection of these two lists.
# Each list consists of disjoint intervals sorted on their start time.
def intersectInterverls(arr1, arr2):
    inter1 = []
    inter2 = []
    for i in arr1:
        inter1.append(Interval(i[0], i[1]))
    for i in arr2:
        inter2.append(Interval(i[0], i[1]))
    i = 0
    j = 0
    output = []
    while i < len(arr1) and j < len(arr2):
        itv1 = inter1[i]
        itv2 = inter2[j]
        if itv1.start <= itv2.end and itv2.start <= itv1.end:
            start = max(itv1.start, itv2.start)
            end = min(itv1.end, itv2.end)
            output.append(Interval(start, end))

        if itv1.end < itv2.end:
            i += 1
        else:
            j += 1
    for i in output:
        print(i.start, i.end)


# arr1=[[1, 3], [5, 6], [7, 9]]
# arr2=[[2, 3], [5, 7]]
# intersectInterverls(arr1,arr2)


# Given a list of intervals representing the start and end time of ‘N’ meetings,
# find the minimum number of rooms required to hold all the meetings.

class Meeting:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __lt__(self, other):
        return self.end < other.end


def minRoomRequired(meetings):
    meets = []
    for meet in meetings:
        meets.append(Meeting(meet[0], meet[1]))

    meets = sorted(meets, key=lambda x: x.start)
    activeMeets = []
    minRoom = 0
    for meeting in meets:
        if len(activeMeets) > 0 and meeting.start >= activeMeets[0].end:
            heapq.heappop(activeMeets)
        heapq.heappush(activeMeets, meeting)
        minRoom = max(minRoom, len(activeMeets))
    print(minRoom)


# meetings=[[1,4], [2,3], [3,6]]
# minRoomRequired(meetings)

class CPULoad:
    def __init__(self, start, end, load):
        self.start = start
        self.end = end
        self.load = load

    def __lt__(self, other):
        return self.end < other.end


def maxCpuLoad(arr):
    jobs = []
    for job in arr:
        jobs.append(CPULoad(job[0], job[1], job[2]))
    jobs = sorted(jobs, key=lambda x: x.start)
    activeJobs = []
    maxLoad = 0
    currLoad = 0
    for job in jobs:
        while len(activeJobs) > 0 and job.start >= activeJobs[0].end:
            currLoad -= activeJobs[0].load
            heapq.heappop(activeJobs)
        heapq.heappush(activeJobs, job)
        currLoad += job.load
        maxLoad = max(maxLoad, currLoad)
    print(maxLoad)


# arr=[[1,4,2], [2,4,1], [3,6,5]]
# maxCpuLoad(arr)

# For ‘K’ employees, we are given a list of intervals representing the working hours of each employee.
# Our goal is to find out if there is a free interval that is common to all employees.
# You can assume that each list of employee working hours is sorted on the start time.

class EmployeeInterval:
    def __init__(self, interval, employeeIndex, intervalIndex):
        self.interval = interval
        self.employeeIndex = employeeIndex
        self.intervalIndex = intervalIndex

    def __lt__(self, other):
        return self.interval.start < other.interval.start


def findFreeSlot(schedule):
    minheap = []
    freeSlot = []
    for i in range(len(schedule)):
        heapq.heappush(minheap, EmployeeInterval(Interval(schedule[i][0][0], schedule[i][0][1]), i, 0))

    prevInterval = minheap[0].interval
    while len(minheap) > 0:
        currEmployee = heapq.heappop(minheap)
        if currEmployee.interval.start > prevInterval.end:
            freeSlot.append(Interval(prevInterval.end, currEmployee.interval.start))
            prevInterval = currEmployee.interval
        else:
            prevInterval.end = max(prevInterval.end, currEmployee.interval.end)
        index = currEmployee.employeeIndex
        if len(schedule[index]) > currEmployee.intervalIndex + 1:
            heapq.heappush(minheap, EmployeeInterval(Interval(schedule[index][currEmployee.intervalIndex + 1][0],
                                                              schedule[index][currEmployee.intervalIndex + 1][1]),
                                                     index, currEmployee.intervalIndex + 1))
    for i in freeSlot:
        print(i.start, i.end)


Schedule = [[[1, 3], [5, 6]], [[2, 3], [6, 8]]]
findFreeSlot(Schedule)


