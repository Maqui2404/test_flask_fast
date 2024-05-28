from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objs as go
import json
from sklearn.metrics import cohen_kappa_score
import scipy.stats as stats
from statsmodels.stats.inter_rater import fleiss_kappa
from statsmodels.stats.contingency_tables import cochrans_q

app = Flask(__name__)
df = None

# Statistical test functions
def one_sample_z_test(sample, pop_mean):
    sample_mean = np.mean(sample)
    n = len(sample)
    pop_std = np.std(sample, ddof=1)
    if pop_std == 0:
        return None, None
    z = (sample_mean - pop_mean) / (pop_std / np.sqrt(n))
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value

def two_sample_z_test(sample1, sample2):
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    n1, n2 = len(sample1), len(sample2)
    std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
    pooled_std = np.sqrt((std1**2 / n1) + (std2**2 / n2))
    if pooled_std == 0:
        return None, None
    z = (mean1 - mean2) / pooled_std
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value

def z_test_for_proportions(p1, p2, n1, n2):
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    if p * (1 - p) == 0:
        return None, None
    z = (p1 - p2) / np.sqrt(p * (1 - p) * (1/n1 + 1/n2))
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value

def paired_z_test(sample1, sample2):
    differences = np.array(sample1) - np.array(sample2)
    mean_diff = np.mean(differences)
    n = len(differences)
    pop_std = np.std(differences, ddof=1)
    if pop_std == 0:
        return None, None
    z = mean_diff / (pop_std / np.sqrt(n))
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p_value

def mann_whitney_u_test(sample1, sample2):
    try:
        u_statistic, p_value = stats.mannwhitneyu(sample1, sample2, alternative='two-sided')
        return u_statistic, p_value
    except Exception as e:
        return None, str(e)

def fisher_exact_test(table):
    try:
        odds_ratio, p_value = stats.fisher_exact(table)
        return odds_ratio, p_value
    except Exception as e:
        return None, str(e)

def fisher_f_test(sample1, sample2):
    try:
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
        f_statistic = var1 / var2
        dfn, dfd = len(sample1) - 1, len(sample2) - 1
        p_value = 1 - stats.f.cdf(f_statistic, dfn, dfd)
        return f_statistic, p_value
    except Exception as e:
        return None, str(e)

def kendall_tau_test(sample1, sample2):
    try:
        tau, p_value = stats.kendalltau(sample1, sample2)
        return tau, p_value
    except Exception as e:
        return None, str(e)

def cochran_q_test(df):
    try:
        q_statistic, p_value = cochrans_q(df)
        return q_statistic, p_value
    except Exception as e:
        return None, str(e)

def cohen_kappa_test(rater1, rater2):
    try:
        kappa = cohen_kappa_score(rater1, rater2)
        return kappa, None
    except Exception as e:
        return None, str(e)

def fleiss_kappa_test(df):
    try:
        kappa = fleiss_kappa(df, method='fleiss')
        return kappa, None
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global df
    file = request.files['csv_file']
    df = pd.read_csv(file)
    return redirect(url_for('summary'))

@app.route('/summary')
def summary():
    global df

    # Calculate summary statistics
    summary_stats = df.describe(include=[np.number]).to_dict()

    # Additional statistics
    mode = df.mode().iloc[0].to_dict()
    for column in df.select_dtypes(include=[np.number]).columns:
        summary_stats[column]['mode'] = mode[column]
        summary_stats[column]['q1'] = df[column].quantile(0.25)
        summary_stats[column]['q3'] = df[column].quantile(0.75)

    # Generate plots
    plots = {}
    for column in df.select_dtypes(include=[np.number]).columns:
        # Box plot
        box_fig = px.box(df, y=column, title=f'Box Plot of {column}')
        box_graph = json.dumps(box_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Bar plot
        bar_fig = px.histogram(df, x=column, title=f'Bar Plot of {column}')
        bar_graph = json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder)

        plots[column] = {'box': box_graph, 'bar': bar_graph}

    return render_template('summary.html', summary_stats=summary_stats, plots=plots, columns=df.columns)

@app.route('/choose_test', methods=['POST'])
def choose_test():
    global df
    test_type = request.form['test_type']
    column1 = request.form.get('column1')
    column2 = request.form.get('column2')
    columns = request.form.getlist('columns')
    pop_mean = request.form.get('pop_mean')

    if test_type == '1':
        sample = df[column1].dropna().values
        pop_mean = float(pop_mean)
        z, p = one_sample_z_test(sample, pop_mean)
        if z is None or p is None:
            return render_template('result.html', z="NaN", p="NaN", error="No se puede calcular el estadístico Z con desviación estándar 0 o datos constantes")
    elif test_type == '2':
        sample1 = df[column1].dropna().values
        sample2 = df[column2].dropna().values
        z, p = two_sample_z_test(sample1, sample2)
        if z is None or p is None:
            return render_template('result.html', z="NaN", p="NaN", error="No se puede calcular el estadístico Z con desviación estándar 0 o datos constantes")
    elif test_type == '3':
        p1 = df[column1].mean()
        p2 = df[column2].mean()
        n1 = df[column1].count()
        n2 = df[column2].count()
        z, p = z_test_for_proportions(p1, p2, n1, n2)
        if z is None or p is None:
            return render_template('result.html', z="NaN", p="NaN", error="No se puede calcular el estadístico Z con error estándar 0")
    elif test_type == '4':
        sample1 = df[column1].dropna().values
        sample2 = df[column2].dropna().values
        z, p = paired_z_test(sample1, sample2)
        if z is None or p is None:
            return render_template('result.html', z="NaN", p="NaN", error="No se puede calcular el estadístico Z con desviación estándar 0 o datos constantes")
    elif test_type == '5':
        sample1 = df[column1].dropna().values
        sample2 = df[column2].dropna().values
        z, p = mann_whitney_u_test(sample1, sample2)
    elif test_type == '6':
        table = pd.crosstab(df[column1], df[column2])
        if table.shape == (2, 2):
            z, p = fisher_exact_test(table)
        else:
            return render_template('result.html', z="NaN", p="NaN", error="La tabla de contingencia no es 2x2. La prueba exacta de Fisher solo es válida para tablas 2x2")
    elif test_type == '7':
        sample1 = df[column1].dropna().values
        sample2 = df[column2].dropna().values
        z, p = fisher_f_test(sample1, sample2)
    elif test_type == '8':
        sample1 = df[column1].dropna().values
        sample2 = df[column2].dropna().values
        z, p = kendall_tau_test(sample1, sample2)
    elif test_type == '9':
        data = df[columns].dropna()
        z, p = cochran_q_test(data)
    elif test_type == '10':
        rater1 = df[column1].dropna().values
        rater2 = df[column2].dropna().values
        z, p = cohen_kappa_test(rater1, rater2)
    elif test_type == '11':
        data = df[columns].dropna()
        z, p = fleiss_kappa_test(data)
    else:
        return render_template('result.html', z="Invalid", p="Invalid", error="Invalid test type selected.")

    return render_template('result.html', z=z, p=p, error=None)

if __name__ == '__main__':
    app.run(debug=True)

#  Running on http://127.0.0.1:5000