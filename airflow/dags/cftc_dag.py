from airflow.models import DAG
from airflow.operators import DummyOperator, BashOperator, PythonOperator
from datetime import datetime, timedelta

execfile("startup.py")
from qfl.etl.data_ingest import *

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2016, 8, 26),
    'email': ['beifert@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
}

# The execution date as YYYY-MM-DD
date = "{{ ds }}"

dag = DAG('cftc_weekly',
          start_date=datetime(2016, 8, 26),
          schedule_interval="0 20 * * FRI",
          default_args=default_args)

t1 = PythonOperator(task_id='weekly_cftc_commodities_ingest',
                    python_callable=WeeklyCftcCommodityIngest.launch,
                    dag=dag,
                    provide_context=True)

t2 = PythonOperator(task_id='weekly_cftc_financials_ingest',
                    python_callable=WeeklyCftcFinancialsIngest.launch,
                    dag=dag,
                    provide_context=True)

run_this_last = DummyOperator(task_id='run_this_last', dag=dag)

run_this_last.set_upstream(t1)
