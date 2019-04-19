# text analytics
unzip folder resources
```
cat resources.zipa* > resources.zip
unzip resources.zip
rm -f resources.zip*
```


model wt103
```
/content/simple-sentiment/data/aclImdb/models/wt103/   
wget http://files.fast.ai/models/wt103/bwd_wt103.h5   
wget http://files.fast.ai/models/wt103/bwd_wt103_enc.h5    
wget http://files.fast.ai/models/wt103/fwd_wt103.h5   
wget http://files.fast.ai/models/wt103/fwd_wt103_enc.h5    
wget http://files.fast.ai/models/wt103/itos_wt103.pkl   
```

google cloud gpu
https://cloud.google.com/compute/docs/gpus/add-gpus#install-driver-script

Demo preview   
![text analytics](https://raw.githubusercontent.com/votamvan/text-analytics/master/images/demo01.png)
