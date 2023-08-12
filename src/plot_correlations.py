import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


trace_link_mean = np.array([50.0, 70.6, 80.6, 88.6, 91.1, 97.2, 99.1, 99.1, 98.4, 98.7, 98.3, 97.1])
trace_anomaly_mean = np.array([50.0, 77.6, 67.9, 69.4, 80.8, 86.4, 88.0, 85.3, 85.5, 85.1, 86.0, 87.6])
trace_link_std = np.array([0, 0.4, 7.7, 0.8, 1.1, 0.3, 0.1, 0.1, 0.2, 0.3, 0.1, 0.3])
trace_anomaly_std = np.array([0, 1.3, 14.5, 6.6, 0.7, 1.9, 1.2, 1.5, 1.4, 1.0, 0.5, 0.8])
theia_link_mean = np.array([50.0, 70.7, 69.1, 71.9, 88.2, 92.4, 96.4, 96.9, 96.9, 96.6, 96.8, 91.6])
theia_anomaly_mean = np.array([50.0, 66.3, 51.0, 51.5, 73.6, 80.3, 85.0, 83.5, 86.9, 83.3, 85.4, 77.1])
theia_link_std = np.array([0, 0.4, 4.3, 1.8, 0.6, 1.7, 0.2, 0.1, 0.2, 0.2, 0.3, 0.5])
theia_anomaly_std = np.array([0, 2.6, 12.8, 16.5, 1.2, 2.8, 2.2, 1.0, 2.5, 1.9, 2.8, 4.6])

theia_sort_idx = np.argsort(theia_link_mean)
trace_sort_idx = np.argsort(trace_link_mean)

trace_link_mean = trace_link_mean[trace_sort_idx]
theia_link_mean = theia_link_mean[theia_sort_idx]
trace_anomaly_mean = trace_anomaly_mean[trace_sort_idx]
theia_anomaly_mean = theia_anomaly_mean[theia_sort_idx]

model_names = ['MLP_no_feats', 'MLP', 'GAT', 'RGCN', 'TGN_no_feats_no_mem', 'TGN_no_feats', 'TGN_no_mem','TGN', '1-hot-dir-TGN', 'Dir-TGN', 'Hetero-TGN', 'HGT-TGN']
assert len(model_names) == len(trace_link_mean) == len(trace_link_std) == len(theia_link_mean) == len(theia_link_std) == len(trace_anomaly_mean) == len(trace_anomaly_std) == len(theia_anomaly_mean) == len(theia_anomaly_std)

fig= plt.figure()
plt.rcParams['lines.markersize'] = 8
plt.rcParams['axes.axisbelow'] = True
r_trace = pearsonr(trace_link_mean, trace_anomaly_mean)
r_theia = pearsonr(theia_link_mean, theia_anomaly_mean)
plt.rcParams.update({'font.size': 14})
plt.grid()
plt.scatter(x = trace_link_mean[:4], y = trace_anomaly_mean[:4], c='#785ef0', marker='v')
plt.scatter(x = theia_link_mean[:4], y = theia_anomaly_mean[:4], c='#ffb000', marker='v')
plt.scatter(x = trace_link_mean[4:], y = trace_anomaly_mean[4:], c='#785ef0', marker='x')
plt.scatter(x = theia_link_mean[4:], y = theia_anomaly_mean[4:], c='#ffb000', marker='x')

markers = ['v', 'x'] 
colors = ['#785ef0', '#ffb000']
labels = [f'TRACE, r = {r_trace.statistic:.2f}' + r" ($\rho$" + f' = {r_trace.pvalue:.0e})', f'THEIA, r = {r_theia.statistic:.2f}' + r" ($\rho$" + f' = {r_theia.pvalue:.0e})', 'Static baselines', 'TGN-based'] 

f = lambda m,c: plt.plot([],[], marker = m, color=c, ls="none")[0]
g = lambda m,c: plt.plot([],[], ls = m, color=c)[0]

handles = [g("dashed", colors[i]) for i in range(2)]
handles += [f(markers[i], "k") for i in range(2)]

coef_trace = np.polyfit(trace_link_mean, trace_anomaly_mean, 1)
poly1d_fn_trace = np.poly1d(coef_trace)

coef_theia = np.polyfit(theia_link_mean, theia_anomaly_mean, 1)
poly1d_fn_theia = np.poly1d(coef_theia) 
# poly1d_fn is now a function which takes in x and returns an estimate for y

space = 3
dash_len = 3
trace_link_mean = np.concatenate((np.zeros(1), trace_link_mean, np.ones(1)*100))
theia_link_mean = np.concatenate((np.zeros(1), theia_link_mean, np.ones(1)*100))
plt.plot(trace_link_mean, poly1d_fn_trace(trace_link_mean), '--', c=colors[0], dashes=(dash_len, space))
plt.plot(theia_link_mean, poly1d_fn_theia(theia_link_mean), '--', c=colors[1], dashes=(dash_len, space))
plt.legend(handles, labels, loc='upper left', framealpha=1)
#plt.xticks(rotation = 45)
plt.xlim(45, 100)
plt.ylim(45, 100)
plt.xlabel("$AuC_{LP}$")
plt.ylabel("$AuC_{AD}$")
size = fig.get_size_inches()
print(size)
plt.savefig("figures/anom_correlation.pdf", bbox_inches='tight')