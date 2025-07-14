---
layout: post
title:  "etcd 服务注册与发现内部 keeplived 原理"
date:   2025-07-14 10:24:00 +0800
tags:
  - Golang
---

在 etcd 服务注册方 KeepAlive 机制中，主要是用一个 channel 接收 keepalive 响应, 其原理如下：

## 工作原理
1. 租约创建：
```
leaseResp, err := cli.Grant(ctx, s.EtcdTTL)
创建一个带 TTL 的租约，如果不续约会自动过期。
```

2. 开启 KeepAlive：
```
ch, err := cli.KeepAlive(ctx, leaseResp.ID)
KeepAlive 返回一个 channel，etcd 客户端会：
自动定期发送 keepalive 请求到 etcd 服务器, 看v3 代码它用的是grpc
默认间隔 是租约 TTL 的 1/3
每次收到服务器响应后，通过 ch 发送 LeaseKeepAliveResponse
```
3. 响应处理：
```
go func() {
    for chResp := range ch {
        s.L.Debug("续约：", logger.String("resp", chResp.String()))
    }
}()
```

### 内部机制
```
- 客户端自动续约：etcd 客户端库在后台启动 goroutine 定期发送续约请求
- 服务器确认：etcd 服务器收到请求后重置租约 TTL，并返回响应
- Channel 通知：客户端收到响应后将 LeaseKeepAliveResponse 发送到 ch
- 异常处理：如果网络断开或服务器异常，channel 会被关闭，for range 循环退出
```

### etcd服务消费者的心跳检测
```
调用方（服务消费者）与 etcd 的心跳检测使用 Watch 机制：
// 调用方监听服务变化
watchChan := client.Watch(ctx, "service/user-service", clientv3.WithPrefix())
for watchResp := range watchChan {
    for _, event := range watchResp.Events {
        switch event.Type {
        case mvccpb.PUT:
            // 服务上线
        case mvccpb.DELETE:
            // 服务下线（租约过期）
        }
    }
}
```
