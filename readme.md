# Crack Tracking
裂纹追踪系统

## tips

- 最快支持2s一张图片输入
- 输出坐标时间约为4-6s
- 使用main.py来开始
- 每次开始之前试验机坐标应当清零
- 已知支持1080*1920 .JPG格式输入，其他格式未做测试
- 裂纹生长暂时~~只支持左右两个方向~~支持上下左右四个方向
- 生成release的时候要加入image、config.ini、model、logs
- 试验机左下角为(0，0)，移动的应当是裂纹而不是摄像头

## v0.00
    First
    
## v0.01
坐标以绝对值而不是相对值的形式传送   
为打包发布做了一些准备，包括IMPORT的精简(吐了，它为啥还这么大啊    
调试了TcpIP通讯  
修复了若干bug

## v0.02
修复了若干bug    
第一个release上线啦！！    
添加了requirements

## v0.03
修复了若干bug    
增加了上下两个裂纹生长方向   
~~修改了裂纹尖端的判断逻辑，从最大连通域改为去噪，暂定Area<Area_max/8的连通域为噪点（效果不错！~~

## v0.04
修复了若干bug    
封装了整个tkinter作为一个class(鸣谢tyf，呜呜呜)    
回滚了裂纹尖端的判断逻辑，即最大连通域的端点，去噪方法解决极端问题有效，在一般问题中失效(可能是去噪阈值还需要再调整     
添加了日志系统 将所有抛出的错误写入日志，目的是检测闪退问题  
怀疑程序占用内存溢出被kill(非常合理)，检查了尽可能多的内存问题(问题竟在plt.close    
