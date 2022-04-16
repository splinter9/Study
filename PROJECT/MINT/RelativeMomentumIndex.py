# 상대모멘텀지수

# //@version=3
// Copyright (c) 2018-present, Alex Orekhov (everget)
// Relative Momentum Index script may be freely distributed under the MIT license.
study("Relative Momentum Index", shorttitle="RMI")

length = input(title="Length", type=integer, minval=1, defval=14)
momentumLength = input(title="Momentum Length", type=integer, minval=1, defval=3)
highlightBreakouts = input(title="Highlight Overbought/Oversold Breakouts ?", type=bool, defval=true)
src = input(title="Source", type=source, defval=close)

up = rma(max(change(src, momentumLength), 0), length)
down = rma(-min(change(src, momentumLength), 0), length)

rmi = down == 0 ? 100 : up == 0 ? 0 : 100 - (100 / (1 + up / down))

obLevel = 70
osLevel = 30

rmiColor = rmi > obLevel ? #0ebb23 : rmi < osLevel ? #ff0000 : #f4b77d
plot(rmi, title="RMI", linewidth=2, color=rmiColor, transp=0)

transparent = color(white, 100)

maxLevelPlot = hline(100, title="Max Level", linestyle=dotted, color=transparent)
obLevelPlot = hline(obLevel, title="Overbought Level", linestyle=dotted)
hline(50, title="Middle Level", linestyle=dotted)
osLevelPlot = hline(osLevel, title="Oversold Level", linestyle=dotted)
minLevelPlot = hline(0, title="Min Level", linestyle=dotted, color=transparent)

fill(obLevelPlot, osLevelPlot, color=purple, transp=95)

obFillColor = rmi > obLevel and highlightBreakouts ? green : transparent
osFillColor = rmi < osLevel and highlightBreakouts ? red : transparent

fill(maxLevelPlot, obLevelPlot, color=obFillColor, transp=90)
fill(minLevelPlot, osLevelPlot, color=osFillColor, transp=90)
