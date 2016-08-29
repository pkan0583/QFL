from airflow.models import DAG
from airflow.operators import DummyOperator, BashOperator, PythonOperator
from datetime import datetime, timedelta

execfile("startup.py")
from qfl.etl.data_ingest import *

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
          schedule_interval="0 8,20 * * MON-FRI",
          default_args=default_args)

t1 = PythonOperator(task_id='daily_futures_price_ingest',
                    python_callable=DailyFuturesPriceIngest.launch,
                    dag=dag,
                    provide_context=True)

t2 = PythonOperator(task_id='daily_generic_futures_price_ingest',
                    python_callable=DailyGenericFuturesPriceIngest.launch,
                    dag=dag,
                    provide_context=True)

t3 = PythonOperator(task_id='daily_equity_index_price_ingest',
                    python_callable=DailyEquityIndexPriceIngest.launch,
                    dag=dag,
                    op_kwargs={'date': date},
                    provide_context=True)

t4 = PythonOperator(task_id='daily_orats_data_ingest',
                    python_callable=DailyOratsIngest.launch,
                    dag=dag,
                    op_kwargs={'date': date},
                    provide_context=True)

t5 = PythonOperator(task_id='daily_equity_price_ingest',
                    python_callable=DailyEquityPriceIngest.launch,
                    dag=dag,
                    provide_context=True)

t6 = PythonOperator(task_id='daily_optionworks_ingest',
                    python_callable=DailyOptionWorksIngest.launch,
                    dag=dag,
                    provide_context=True)

run_this_last = DummyOperator(task_id='run_this_last', dag=dag)

t2.set_upstream(t1)
# t3.set_upstream(t2)
# t4.set_upstream(t3)
# t5.set_upstream(t4)
run_this_last.set_upstream(t4)
