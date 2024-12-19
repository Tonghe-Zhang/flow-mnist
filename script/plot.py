
import numpy as np
import matplotlib.pyplot as plt

denoising_steps= [1, 2, 4, 8, 16, 20, 32, 64, 128, 256, 512]

shortcut_no_balance = [
    1209.1765762253845,
    1647.0992399761221,
    1696.6279999949247,
    1379.0920392713222,
    1034.6627804605557,
    990.2904957349697,
    844.2398463105458,
    698.5943685711503,
    704.5023614492458,
    646.9494690203223,
    670.1949565141392
]
# Define the colors
light_blue = '#ADD8E6'
medium_blue = '#0000FF'
dark_blue = '#00008B'

no_shortcut=np.load('/home/zhangtonghe/flow-mnist/eval/ReFlow/24-12-14-17-41-48/eval_data.npz')['fid_scores']
# [976.18135186,582.43485619,403.29071627,409.05113879,343.43643976,376.0732125,396.9363207,413.86088656,428.51481331,453.66004748,483.94108714]


shortcut_balance_linear=np.load('/home/zhangtonghe/flow-mnist/eval/ShortCutReFlow/24-12-14-17-42-39/eval_data.npz')['fid_scores']
#[710.48207456,585.94382846,476.10481659,458.33951881,405.63448317,420.73385388,357.0016595,,374.74648381,385.2026475,,397.8379337,401.02911742]


shortcut_balance_always=np.load('/home/zhangtonghe/flow-mnist/eval/ShortCutReFlow/24-12-14-17-43-16/eval_data.npz')['fid_scores']
#[836.86639132,662.61128674,563.4689078,,502.8253775,,467.0907912,392.37057261,375.03518748,334.18752086,327.97690281,312.39245341,373.85175542]


# Create the plot
plt.figure(figsize=(16, 9))

# Plot each list
plt.semilogx(denoising_steps, shortcut_no_balance, label='Shortcut: 1:1', marker='o', color='purple')
plt.semilogx(denoising_steps, shortcut_balance_always, label='Shortcut: always balance', marker='s', color=dark_blue)
plt.semilogx(denoising_steps, shortcut_balance_linear, label='Shortcut: linear balance', marker='d', color=medium_blue)
plt.semilogx(denoising_steps, no_shortcut, label='No Shortcut', marker='^', color='red')

# Enlarge the legend
plt.legend(fontsize=20)

# Grid on
plt.grid(True)

# Enlarge the tick labels
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Labels for the axes
plt.xlabel('Denoising Steps', fontsize=17)
plt.ylabel('FID Scores', fontsize=17)

# Save the plot as a PDF
plt.savefig('fid_scores_plot.pdf', bbox_inches='tight')

# Show the plot
plt.show()