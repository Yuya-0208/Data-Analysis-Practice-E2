import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from IPython.display import display, Markdown

def display_title(s, pref='Figure', num=1, center=False):
    ctag = 'center' if center else 'p'
    s = f'<{ctag}><span style="font-size: 1.2em;"><b>{pref} {num}</b>: {s}</span></{ctag}>'
    if pref == 'Figure':
        s = f'{s}<br><br>'
    else:
        s = f'<br><br>{s}'
    display(Markdown(s))

def central(x):
    x0 = np.mean(x)
    x1 = np.median(x)
    x2 = stats.mode(x, keepdims=True)[0] 
    if isinstance(x2, np.ndarray):
         x2 = x2[0] 
    return x0, x1, x2


def dispersion(x):
    y0 = np.std(x)  
    y1 = np.min(x)  
    y2 = np.max(x)  
    y3 = y2 - y1  
    y4 = np.percentile(x, 25)  
    y5 = np.percentile(x, 75) 
    y6 = y5 - y4 
    return y0, y1, y2, y3, y4, y5, y6

def display_central_tendency_table(df, num=1, round_digits=6):
    display_title('Central tendency summary statistics.', pref='Table', num=num, center=False)
    df_central = df.apply(central, axis=0, result_type='expand') 
    df_central = df_central.round(round_digits)
    row_labels = 'mean', 'median', 'mode'
    df_central.index = row_labels
    display(df_central)


def display_dispersion_table(df, num=1, round_digits=6):
    display_title('Dispersion summary statistics.', pref='Table', num=num, center=False)
    df_dispersion = df.apply(dispersion, axis=0, result_type='expand').round(round_digits)
    row_labels_dispersion = 'st.dev.', 'min', 'max', 'range', '25th', '75th', 'IQR'
    df_dispersion.index = row_labels_dispersion
    display(df_dispersion)

def corrcoeff(x, y):
    return np.corrcoef(x, y)[0, 1]


def plot_regression_line(ax, x, y, **kwargs):
    a, b = np.polyfit(x, y, deg=1)
    x0, x1 = min(x), max(x)
    y0, y1 = a * x0 + b, a * x1 + b
    ax.plot([x0, x1], [y0, y1], **kwargs)

def plot_descriptive(df, target_col='Deli', transform=False, split_by_target=False):
    
    
    
    y = df[target_col]
    ivs = [df['Fre'], df['Milk'], df['Gro'], df['Fro']]
    colors = 'b', 'r', 'g', 'y'
    xlabels = 'Fresh', 'Milk', 'Grocery', 'Frozen'
    
    if transform:
        ivs = [np.around(x / 1000, 1) for x in ivs]
        xlabels = [s + '1' for s in xlabels] 
        
    fig, axs = plt.subplots(2, 2, figsize=(8, 6), tight_layout=True)
    axs_flat = axs.ravel()
    
    for i, (ax, x, c, s) in enumerate(zip(axs_flat[:3], ivs, colors, xlabels)):
        ax.scatter(x, y, alpha=0.5, color=c)
        plot_regression_line(ax, x, y, color='k', ls='-', lw=2)
        
        r = corrcoeff(x, y)
        ax.text(0.7, 0.3, f'r = {r:.3f}', color=c, 
                transform=ax.transAxes, bbox=dict(color='0.8', alpha=0.7))
        ax.set_xlabel(s)

    axs[0, 1].set_xticks([25000, 50000, 75000, 100000] if not transform else [25, 50, 75, 100])
    [ax.set_ylabel(target_col) for ax in axs[:, 0]]
    [ax.set_yticklabels([]) for ax in axs[:, 1]]

    ax = axs[1, 1]
    
    if split_by_target:  
        split_value = y.median() 
        i_low = y <= split_value
        i_high = y > split_value
        
     
        fre_col = df['Fre'] if not transform else ivs[0]

        fcolors = 'm', 'c'
        labels = 'Lower-expense', 'Higher-expense'
        q_groups = [[200, 600, 1000], [1000, 11000, 21000]] 
        ylocs = 0.3, 0.7
        
        for i, c, s, qs, yloc in zip([i_low, i_high], fcolors, labels, q_groups, ylocs):
            ax.scatter(fre_col[i], y[i], alpha=0.5, color=c, facecolor=c, label=s)
            plot_regression_line(ax, fre_col[i], y[i], color=c, ls='-', lw=2)
            
            
            [ax.plot(fre_col[i].mean(), q, 'o', color=c, mfc='w', ms=10) for q in qs]
            
            r = corrcoeff(fre_col[i], y[i])
            ax.text(0.7, yloc, f'r = {r:.3f}', color=c, 
                    transform=ax.transAxes, bbox=dict(color='0.8', alpha=0.7))

        ax.legend()
        ax.set_xlabel('Fresh' if not transform else 'Fresh1')
    
    
    panel_labels = 'a', 'b', 'c', 'd'
    [ax.text(0.02, 0.92, f'({s})', size=12, transform=ax.transAxes) 
     for ax, s in zip(axs_flat, panel_labels)]
    
    plt.show()
    display_title('Correlations amongst main variables.', pref='Figure', num=1)