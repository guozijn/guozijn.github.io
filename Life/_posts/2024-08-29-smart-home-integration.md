---
title: "Apple HomeKit 智能家居低成本入门指南"
tags:
  - homekit
  - home assistant
  - smart home
---

> 本文记录了我如何用低成本打造一个 **苹果 HomeKit 智能家居系统** 的全过程，从选购硬件到软件配置，以及最终的使用体验。希望能给同样想入坑的朋友一些参考。

## 一、起因

卧室吸顶灯坏了，想着先换个 LED 灯芯试试是不是灯芯的问题。于是花了约 ¥40 买了一款支持米家 WiFi 接入的灯芯。  
装好后才发现买的是 48W 的灯条，亮度一般，甚至不如台灯。不过既然能亮，就先凑合用。

这让我萌生了一个想法：既然要折腾，不如顺便把整个家里的灯光和电器做一套智能化改造。家里正好有个闲置的 **HomePod mini**，于是决定以 HomeKit 为核心做实验。

## 二、思路与方案

### 1. 跨生态问题

苹果 HomeKit 和米家、涂鸦等生态协议不兼容。  
解决方案是使用 **Home Assistant（HA）** —— 一款开源智能家居网关软件，能桥接不同协议，把米家、涂鸦等设备“伪装”成 HomeKit 设备，从而让 HomeKit 统一管理。

### 2. 硬件准备

- **运行环境**：树莓派太贵，我在二手市场买了一个“智趣盒子”（¥75，2 GB RAM + 16 GB ROM），刷机后运行 HA。  
- **网络环境**：主路由在客厅，卧室信号差，于是用卧室网口接了一个桥接路由器，并接入盒子。这样手机能在全屋无感切换 WiFi。  
- **更新与调试**：刷好机后访问 `http://192.168.31.31:8123/`，发现 HA 无法自动更新。通过 SSH 登录查看，发现是 Docker 部署，于是手动拉取最新版镜像并重启，问题解决。

**Home Assistant Docker Services**

![Home Assistant Docker services](https://images.zijianguo.com/sfbox01.png){:width="90%"}

## 三、设备接入

### 1. 米家设备

接入流程：

1. 确认有 **家庭中枢**（HomePod / iPad），才能实现远程控制。
2. 在 HACS 安装插件 **Xiaomi Miot Auto**。

    ![Xiaomi Miot Auto](https://images.zijianguo.com/ha03.png){:width="90%"}
3. 添加米家账号并导入设备。

    ![Device and Service](https://images.zijianguo.com/ha05.png){:width="90%"}
    ![Miiot](https://images.zijianguo.com/miiot01.png){:width="90%"}

4. 创建 **HomeKit Bridge** 并桥接至 HomeKit。
   - 注意：一个桥接中只能有一个空调，所以我把卧室空调单独建桥。

    ![hb01](https://images.zijianguo.com/hb01.png){:width="90%"}
    ![hb02](https://images.zijianguo.com/hb02.png){:width="90%"}

最终在 iOS “家庭”App 中能直接看到米家设备，并通过按钮或 Siri 控制。

![homeapp](https://images.zijianguo.com/homeapp.jpeg){:width="30%"}

### 2. 涂鸦设备

主要用于红外 / 射频控制的电器，例如电视、晾衣架。

步骤：

1. 在 **涂鸦开发者平台** 创建云项目，配置权限。

    ![ty01](https://images.zijianguo.com/ty01.png){:width="90%"}
    ![ty03](https://images.zijianguo.com/ty03.png){:width="90%"}

2. 在涂鸦或“智能生活”App 中完成设备配对与学习遥控信号。

    ![tyapp01](https://images.zijianguo.com/tyapp01.png){:width="30%"}
    ![tyapp02](https://images.zijianguo.com/tyapp02.png){:width="30%"}

3. 将密钥填入 HA，设备，在HomeKit Bridge中添加对应的开关，即可出现在 HomeKit 中。

    ![ty04](https://images.zijianguo.com/ty04.png){:width="90%"}

### 3. 其他设备接入方案

- **灯具**：  
  - 灯条控制：能调色温，但断电后失效。  
  - 智能开关：手动/智能两用，更推荐。

- **空调**：  
  - IoT 空调可直接接入。  
  - 普通空调需配合 **空调伴侣**（如 Gosund 电小酷）。

- **电视 / 晾衣架**：  
  - 用支持红外/射频的 **万能遥控器**（涂鸦平台）。  

- **插座**：  
  - 用智能插座即可实现通断电控制。  

- **门铃**：  
  - 高端选 `Aqara G4`，性价比选 **小米智能门铃 3**。  

- **窗帘**：  
  - 可选轨道电机或窗帘伴侣（体验一般，我退货了）。

## 四、成本清单

| 项目 | 金额 (¥) |
|------|----------|
| 二手 HomePod mini | 599 |
| 智趣盒子 | 75 |
| HomeKit 2 路开关 | 45 |
| 涂鸦万能遥控 | 152 |
| 72W HomeKit 灯条 | 85 |
| 米家空调伴侣 | 59 |
| 智能插座 | 30 |
| **合计** | **1045** |

## 五、未来扩展

目前系统已能满足大部分日常需求。下一步可考虑：

- 智能门锁  
- 温湿度传感器  
- 扫地机器人、空气传感器  
- 自动化场景（如“回家模式”、温湿度触发）

借助 HA 的插件与自动化功能，苹果生态下的家庭控制已经足够灵活和强大。

## 总结

通过 **¥1045** 的低成本改造，成功将米家、涂鸦等设备整合进苹果 HomeKit，实现了统一控制。整个过程的关键在于 **Home Assistant 的桥接作用**。  

如果你也想入坑，可以先从一两个设备试起，逐步扩展。  
欢迎在评论区交流你的智能家居玩法。 🚀
