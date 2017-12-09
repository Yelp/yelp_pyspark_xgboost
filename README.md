yelp_pyspark_xgboost
======================

Pyspark wrapper for Xgboost4j

# Getting started

Ensure that xgboost4j dependency is available in your maven repository. If not follow the instructions in the [xgboost installation page](http://xgboost.readthedocs.io/en/latest/jvm/) and build xgboost from source.

## Adding to dependencies

For **Maven**:
mvn clean compile install

```xml
<dependencies>
  <dependency>
    <groupId>com.yelp.ads</groupId>
    <artifactId>yelp_pyspark_xgboost</artifactId>
    <version>1.0.0</version>
  </dependency>
</dependencies>
```

## Adding the jar to spark context

```bash
pyspark --jars yelp_pyspark_xgboost.jar
```

## Using the wrapper for model training

```python

from pyspark.mllib.common import _py2java
from pyspark.mllib.common import _to_java_object_rdd

params = {'eta': 0.1}

# Add hyperparameters to params dict
#params.update({key: value})

model = sc._jvm.com.yelp.ads.gbt.TrainwithRDD.train(
    sc._jsc,
    _to_java_object_rdd(data),
    _py2java(sc, params))
```


## Using the wrapper for prediction

```python
from pyspark.mllib.common import _java2py
from pyspark.mllib.common import _to_java_object_rdd

prediction = float(
    _java2py(
        sc,
        sc._jvm.com.yelp.ads.gbt.TrainwithRDD.predict(
            sc._jsc,
            _to_java_object_rdd(rdd),
            model)
    )
```


## License

yelp_pyspark_xgboost is licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
