# 파인스크립트
# 동적 상대강도지수 

study(title="Dynamic RSI",shorttitle = "DyRSI")
//
DZbuy = 0.1
DZsell = 0.1
Period = 5
Lb = 60
//
red=#0ebb23
green=#787B86
//
RSILine = rsi(close,Period)
jh = highest(RSILine,Lb)
jl = lowest(RSILine,Lb)
jc = (wma((jh-jl)*0.5,Period) + wma(jl,Period))
Hiline = jh - jc * DZbuy
Loline = jl + jc * DZsell
R = (4 * RSILine + 3 * RSILine[1] + 2 * RSILine[2] + RSILine[3] ) / 10
//
a = plot(R, title='R', color=black, linewidth=1, transp=0)
b = plot(Hiline, title='Hiline', color=black, linewidth=1, transp=0)
c = plot(Loline, title='Loline', color=black, linewidth=1, transp=0)
plot(jc, title='Jc', color=purple, linewidth=1, transp=50)
//
col_1 = R > Hiline ? red:na
col_2 = R < Loline ? green:na
//
fill(a, b, color=col_1,transp=0)
fill(a, c, color=col_2,transp=0)
