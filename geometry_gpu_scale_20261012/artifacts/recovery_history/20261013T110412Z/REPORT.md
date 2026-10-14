# GPU几何熵扩大训练结果

完成阶段：38/48；625更新终点：19/24。未完成阶段不能计作成功。

两种degree，四臂，三种子；batch4096、125/625更新。G/H仅在GPU评分；4进程GeoSteiner。旧CPU结果仅作历史背景，新旧耗时差包含多项实现变化。

|degree|更新|方法|实际种子数|latest OOD误差均值%|种子SD|平均进程小时|
|---:|---:|---|---:|---:|---:|---:|
|10|125|original|3|0.200848|0.0014104092155193745|0.639|
|10|125|random|3|0.198439|0.0025454032694041027|0.573|
|10|125|entropy|3|0.196229|0.00010334394295828625|0.639|
|10|125|geometry|3|0.194793|0.0015038632332257016|0.688|
|10|625|original|3|0.182262|0.0015108476242915417|3.084|
|10|625|random|3|0.182987|0.00142893889982655|2.352|
|10|625|entropy|3|0.177985|0.0013895688117973445|2.712|
|10|625|geometry|3|0.180194|0.0003390317927568426|2.812|
|30|125|original|2|0.893842|0.0047049585739294755|3.031|
|30|125|random|2|0.882455|0.000619138058654688|2.079|
|30|125|entropy|2|0.883370|0.0013544600930420684|2.271|
|30|125|geometry|1|0.868281|NA|3.326|
|30|625|original|2|0.845091|0.005720252192617546|9.008|
|30|625|random|2|0.836655|0.005110691153156567|6.369|
|30|625|entropy|2|0.837352|0.004303397344608071|7.703|
|30|625|geometry|1|0.840232|NA|9.832|

主比较是625更新的G−original/random/H，共6项，使用99.1667%条件实例区间；125与best为次级描述结果。种子效应及区间见summary.json。GPU训练保留FP32；G数学定义保留，但浮点计算顺序发生变化，不宣称逐位等价。历史测试已见，全部属于探索性研究。

缺失阶段：/remote-home/Guilin/DRO/Codex/geometry_gpu_scale_20261012/artifacts/train/d30_geometry_s101/u125, /remote-home/Guilin/DRO/Codex/geometry_gpu_scale_20261012/artifacts/train/d30_geometry_s101/u625, /remote-home/Guilin/DRO/Codex/geometry_gpu_scale_20261012/artifacts/train/d30_original_s101/u125, /remote-home/Guilin/DRO/Codex/geometry_gpu_scale_20261012/artifacts/train/d30_original_s101/u625, /remote-home/Guilin/DRO/Codex/geometry_gpu_scale_20261012/artifacts/train/d30_random_s101/u125, /remote-home/Guilin/DRO/Codex/geometry_gpu_scale_20261012/artifacts/train/d30_random_s101/u625, /remote-home/Guilin/DRO/Codex/geometry_gpu_scale_20261012/artifacts/train/d30_geometry_s102/u125, /remote-home/Guilin/DRO/Codex/geometry_gpu_scale_20261012/artifacts/train/d30_geometry_s102/u625, /remote-home/Guilin/DRO/Codex/geometry_gpu_scale_20261012/artifacts/train/d30_entropy_s102/u125, /remote-home/Guilin/DRO/Codex/geometry_gpu_scale_20261012/artifacts/train/d30_entropy_s102/u625
