from airflow.models import DAG
from airflow.operators import DummyOperator, BashOperator, PythonOperator
from datetime import datetime, timedelta
import os
import sys

home = os.path.expanduser("~")
local_repo = os.path.join("Documents", "Code", "qfl")
os.chdir(os.path.join(home, local_repo))  # Activate .env

home = os.path.expanduser("~")
local_repo = os.path.join("Documents", "Code", "qfl")
sys.path.append(os.path.join(home, local_repo))  # Activate .env

modules = ["qfl"]
sub_modules =['qfl', 'etl', 'airflow']
for sm in sub_modules:
    modules.append(os.path.join(modules[0], sm))
modules.append(os.path.join("qfl", "etl"))
modules.append(os.path.join("qfl", "core"))
modules.append(os.path.join("qfl", "utilities"))
modules.append(os.path.join("airflow", "dags"))

from qfl.etl.data_ingest import daily_equity_price_ingest,\
                                daily_futures_price_ingest, \
                                daily_generic_futures_price_ingest, \
                                daily_equity_index_price_ingest, \
                                test_process

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2016, 8, 4),
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

dag = DAG('etl_daily',
          start_date=datetime(2016, 8, 4),
          schedule_interval="0 18 * * MON-FRI",
          default_args=default_args)

t1 = PythonOperator(task_id='daily_equity_price_ingest',
                    python_callable=daily_equity_price_ingest,
                    dag=dag,
                    op_kwargs={'date': date},
                    provide_context=True)

t2 = PythonOperator(task_id='daily_futures_price_ingest',
                    python_callable=daily_futures_price_ingest,
                    dag=dag,
                    op_kwargs={'date': date},
                    provide_context=True)

t3 = PythonOperator(task_id='daily_generic_futures_price_ingest',
                    python_callable=daily_generic_futures_price_ingest,
                    dag=dag,
                    provide_context=True)

t4 = PythonOperator(task_id='daily_equity_index_price_ingest',
                    python_callable=daily_equity_index_price_ingest,
                    dag=dag,
                    op_kwargs={'date': date},
                    provide_context=True)

# t5 = PythonOperator(task_id='test_process',
#                     python_callable=test_process,
#                     dag=dag,
#                     op_kwargs={'date': date},
#                     provide_context=True)

run_this_last = DummyOperator(task_id='run_this_last', dag=dag)

t2.set_upstream(t1)
t3.set_upstream(t2)
t4.set_upstream(t3)
run_this_last.set_upstream(t4)
