# customer_churning
An Application to the ”Churn” Problem

The goal is to predict which of the costumers are going to change telecommunication provider through algorithms of machine learning. Eight new features were created based on the given variables and five algorithms were tested, xGBoost, Random Forest, Decision Trees and Multi-Layer Perceptron. The best score was achieved with the xGBoost algorithm.

<b>The extra features we created are the
following</b>:
1) total_minutes: which was obtained by the summation
of the numeric variables ”total day minutes”, ”total
eve minutes”, ”total night minutes” and it represents
the total minutes of the whole day calls.
2) total_calls: which are the summation of the numeric
variables ”total day calls, ”total eve calls”, ”total
night calls” and it is the total calls of the whole day
calls.
3) total_charge: which was derived from the summation
of the numeric variables ”total day charge”,
”total_eve_charge”, ”total night charge. This variable
shows the total charges of the whole day calls.
4) mins_per_call: which is the average number of minutes
per call. It has been calculated by dividing the ”total
minutes” and ”total calls”.
5) charge_per_call: which is the average charge per call. It
has been calculated by dividing the ”total charge” and
”total calls”.
6) mins_per_call_intl: which is the average number of
minutes per international call. It has been calculated by
dividing the ”total intl minutes” and ”total intl calls”.
7) charge_per_call_intl: which is the average charge per
international call. It has been calculated by dividing the
”total_intl_charge” and ”total intl calls”.
8) when_more_calls: which are a categorical value that
corresponds to the period of the day (day, evening or
night) that the customer did the most calls.

<b>Modeling</b>
Various classifiers were tested (xGBoost, Random Forest, Decision Trees and Multi-
Layer Perceptron). xGBoost had the best predictive power in the
given data set and was the best classifier with an accuracy of
0:977. Moreover, it was observed that the new features that were created played an important role in the results since they
improved the accuracy.

Thanks to Vasso Tsichli for the significant help in the project.

Dimosthenis Beleveslis <br>
dimbele4@gmail.com
