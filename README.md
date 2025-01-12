# 批量kill python
```shell
ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9
```
