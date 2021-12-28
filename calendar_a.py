# -*- coding: utf-8 -*-
"""
Write a python program to generate Calendar for given month and year:
(a)By using inbuilt library 
"""
import calendar

y = int(input("Enter year : "))

m = int(input("Enter month : "))

print(calendar.month(y, m))
