get\_topo.py

这个脚本是用于得到topology\_lookup\_table.csv这个映射关系的表格的



///////////////

ANN\_model.py

这个脚本是文献作者的源脚本



///////////////////



ANN\_model\_2.py

这个脚本是我利用作者所说的最优参数所配置的脚本



/////////////////////



ANN\_model\_3.py

这个脚本是用于对合并之后的数据进行训练的脚本，模型保存再ANN\_model\_2



////////////////////////////



ANN\_model\_4.py

这个脚本是以下参数的脚本：

&#x20;   best\_params = {

&#x20;       "batch\_size": 32,

&#x20;       "depth": 4,

&#x20;       "width": 256,

&#x20;       "activation": 'relu',

&#x20;       "alpha": 0.0001,

&#x20;       "optimizer": 'adam',

&#x20;       "early\_stopping": 100,

&#x20;       "dropout": 0.3,

&#x20;       "weight\_decay": 1e-04

&#x20;   }



////////////////////////////////



其中ANN\_models\_2存放着最好的训练参数



/////////////////////////////////



PINN\_model\_1.py

该脚本是用于PINN模型的训练

对应的训练效果存储文件夹在PINN\_models



/////////////////////////////////



PINN\_model\_2.py

该脚本也是PINN的脚本

对应的训练效果存储在PINN\_models\_1

PINN\_models\_2也是基于这个脚本所生成的，不过和前面不同的是这里面保存了scaler.pkl，这是因为一旦你在训练时保存了 scaler.pkl，你的预测脚本就可以彻底脱离庞大的原始训练集，直接使用 scaler = joblib.load('scaler.pkl') 瞬间完成环境加载。



////////////////////////////////////////////



SHAP\_for\_PINN.py

进行shap分析



///////////////////////////////////////

