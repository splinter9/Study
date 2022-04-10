import datetime

year, month, day, n = 2020, 1, 1, 396

day = datetime.date(year, month, day)
theday = day + datetime.timedelta(n)
print(theday)

print('\n')
#2020-1-1
#2020-1-31