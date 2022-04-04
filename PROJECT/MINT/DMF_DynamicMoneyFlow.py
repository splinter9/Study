# //@version=4

# // This source code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/
# // © RezzaHmt

# //...DMF...
study("Dynamic Money Flow", "DMF", format=format.volume, precision=2, resolution="")

mode                = input("Index", "Mode", options=["Index", "Cumulative"])
period              = input(26, "Period", minval=1, tooltip="Only applies to index mode.")
ma_switch           = input("EMA", "Moving Averages", options=["OFF", "EMA", "WMA"], tooltip="Set the type of Moving Averages added on DMF or turn them off. MAs can also be turned off individually by setting the length to zero.")
fast_len            = input(8, "Fast Length", minval=0)
slow_len            = input(20, "Slow Length", minval=0)
simulative_vol      = input(false, "Simulative Volume", group="Experimental Options", tooltip="Use this option if volume is not provided for the security or it's inappropriate.")
vol_power           = input(1.0, "Power", minval=0, maxval=5, group="Experimental Options", tooltip="The power to which volume is raised. Numbers below 1 reduce the significance of volume in calculations, while numbers above 1 add to it. Setting power to zero will exclude volume from calculations.")
weight_distribution = input("Dynamic", "Weight Distribution Method", options=["Dynamic", "Static"], group="Experimental Options", tooltip="This is from where Dynamic Money Flow derives its name. By default, weight distribution is dynamic. However, it can be changed to apply a static bias.")
static_dist_bias    = input(50, "Static Weight Distribution Bias", minval=0, maxval=100, group="Experimental Options", tooltip="After setting the previous option to Static, you can set a bias factor (0 - 100) for weight distribution. When set to zero, only Close to Range comparison will take effect. And with 100, only Close to Close comparison will be applied.")

vol= if simulative_vol
    pow(abs(close - close[1])
      + abs(high - max(open, close)) * 2
      + abs(min(open, close) - low) * 2, vol_power)
    else
    pow(volume, vol_power)
    alpha   = if weight_distribution == "Dynamic"
    tr == 0 ? 0 : abs((close - close[1]) / tr)
    else
    static_dist_bias / 100
    trh     = max(high, close[1])
    trl     = min(low , close[1])
    ctr     = tr == 0 ? 0 :
          ((close - trl) + (close - trh)) / tr * (1 - alpha) * vol
ctc     = close - close[1] == 0 ? 0 :
          close > close[1] ? alpha * vol : -alpha * vol
    dmf     = if mode == "Index"
        rma(ctr + ctc, period) / rma(vol, period)
    else
        cum(ctr + ctc)

fast_ma = if fast_len!=0 
        if ma_switch=="EMA"
        ema(dmf, fast_len)
    else if ma_switch=="WMA"
        wma(dmf, fast_len)
slow_ma = if slow_len!=0
    if ma_switch=="EMA"
        ema(dmf, slow_len)
    else if ma_switch=="WMA"
        wma(dmf, slow_len)

var color main_color = na
if mode == "Index" and dmf > 0
    main_color := #00D8A0
else if  mode == "Index" and dmf < 0
    main_color := #F82060
else if  mode == "Cumulative" and slow_ma and dmf > slow_ma
    main_color := #00D8A0
else if  mode == "Cumulative" and slow_ma and dmf < slow_ma
    main_color := #F82060
else
    main_color := #00D8A0

pMain   = plot(dmf, "DMF", color=main_color, linewidth=2)
pZero   = plot(mode == "Index" ? 0 : na, editable=false, display=display.none)
pFastMA = plot(fast_ma, "Fast MA", color=#08E0A800, linewidth=1)
pSlowMA = plot(slow_ma, "Slow MA", color=#FF286800, linewidth=1)

fill(pMain, pZero, title="Oscillator Background",
      color=iff(dmf > 0, #00D8A006, #F8206006))
fill(pFastMA, pSlowMA, title="Moving Average Fill",
      color=iff(fast_ma > slow_ma, #08E0A858, #FF286858))

hline(mode == "Index" ? 0 : na,    "Zero Line",  color=#787B86FF, linestyle=hline.style_dashed)
hline(mode == "Index" ? 0.1 : na,  "Level",      color=#787B8680, linestyle=hline.style_dashed)
hline(mode == "Index" ? -0.1 : na, "Level",      color=#787B8680, linestyle=hline.style_dashed)
