import os
import pyspark
from pyspark.sql import SQLContext

class spark_init(object):
    def __init__(self, sparkConfig=None, name='ehr'):
        self._setup_spark(sparkConfig)

        self.sc, self.sqlContext = self._init_spark(name=name)

    def _setup_spark(self, sparkConfig):

        if sparkConfig == None:
            config = {'memory': '300g', 'excutors': '4', 'exe_mem': '50G', 'result_size': '80g',
                      'temp': '/home/yikuan/tmp', 'offHeap':'16g'}
        else:
            config = sparkConfig

        os.environ["PYSPARK_PYTHON"] = "" # python spark path
        pyspark_submit_args = ' --driver-memory ' + config['memory'] + ' --num-executors ' + config['excutors'] + \
                              ' --executor-memory ' + config['exe_mem']+ \
                              ' --conf spark.driver.maxResultSize={} --conf spark.memory.offHeap.size={} --conf spark.local.dir={}'\
                                  .format(config['result_size'], config['offHeap'], config['temp']) +\
                              ' pyspark-shell'

        os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args

    def _init_spark(self, name='ehr'):
        sc = pyspark.SparkContext(appName=name)
        sqlContext = SQLContext(sc)
        sqlContext.sql("SET spark.sql.parquet.binaryAsString=true")
        return sc, sqlContext


def read_txt(sc, sqlContext, path):
    """read from txt to pyspark dataframe"""
    file = sc.textFile(path)
    head = file.first()
    content = file.filter(lambda line: line != head).map(lambda k: k.split('\t'))
    df = sqlContext.createDataFrame(content, schema=head.split('\t'))
    return df


def read_parquet(sqlContext, path):
    """read from parquet to pyspark dataframe"""
    return sqlContext.read.parquet(path)

def read_csv(sqlContext, path):
    """read from parquet to pyspark dataframe"""
    return sqlContext.read.csv(path, header=True)