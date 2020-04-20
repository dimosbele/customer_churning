# customer_churning
An Application to the ”Churn” Problem

The goal is to predict which of the costumers are going to change telecommunication provider through algorithms of machine learning. We construct eight new features based on the given variables and we have tested five algorithms, xGBoost, Random Forest, Decision Trees and Multi-Layer Perceptron. The best score is achieved with the xGBoost algorithm, using all the available variables.

The extra features we created are the
following:
1) total_minutes which was obtained by the summation
of the numeric variables ”total day minutes”, ”total
eve minutes”, ”total night minutes” and it represents
the total minutes of the whole day calls.
2) total_calls which are the summation of the numeric
variables ”total day calls, ”total eve calls”, ”total
night calls” and it is the total calls of the whole day
calls.
3) total_charge which was derived from the summation
of the numeric variables ”total day charge”,
”total_eve_charge”, ”total night charge. This variable
shows the total charges of the whole day calls.
4) mins_per_call which is the average number of minutes
per call. It has been calculated by dividing the ”total
minutes” and ”total calls”.
5) charge_per_call which is the average charge per call. It
has been calculated by dividing the ”total charge” and
”total calls”.
6) mins_per_call intl, which is the average number of
minutes per international call. It has been calculated by
dividing the ”total intl minutes” and ”total intl calls”.
7) charge_per_call intl which is the average charge per
international call. It has been calculated by dividing the
”total_intl_charge” and ”total intl calls”.
8) when_more_calls which are a categorical value that
corresponds to the period of the day (day, evening or
night) that the customer did the most calls.
