import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
import matplotlib.colors as mcolors

list_of_color = list(mcolors.TABLEAU_COLORS.values())
figure(figsize=(15, 15), dpi=80)

def plot_epoch_solution(epoch_instance, epoch_solution):
    max_coord = 10000
    min_coord = -100

    plt.clf()
    ax = plt.subplot()

    plt.gca().set_aspect(1, adjustable='box')
    ax.set_xlim([min_coord ,max_coord])
    ax.set_ylim([min_coord ,max_coord])

    ax.scatter(epoch_instance["coords"][0][0],
                epoch_instance["coords"][0][1],
                marker=",", 
                color="k", 
                s=100)

    ax.scatter(epoch_instance["coords"][:,0],
                epoch_instance["coords"][:,1],
                marker="o", 
                color="grey", 
                s=20)


    for color_index, vehicle_route in enumerate(epoch_solution):
        for idx, request_idx in enumerate(vehicle_route):
            real_index = np.where(epoch_instance["request_idx"] == request_idx)[0][0]
            ax.scatter(epoch_instance["coords"][real_index][0],
                           epoch_instance["coords"][real_index][1],
                            marker="o", 
                            color="k", 
                            s=20)
            
            if idx == 0:
                            
                ax.annotate("",
                            xy=(epoch_instance["coords"][real_index][0],
                                epoch_instance["coords"][real_index][1]),
                            xycoords='data',
                            xytext=(epoch_instance["coords"][0][0],
                                    epoch_instance["coords"][0][1]),
                            textcoords='data',
                            arrowprops=dict(arrowstyle="->",
                                            connectionstyle="arc3,rad=0.05",
                                            shrinkA=10, shrinkB=10, color=list_of_color[color_index%len(list_of_color)]),
                            )
            else:     
                previous_real_index = np.where(epoch_instance["request_idx"] == vehicle_route[idx-1])[0][0]
                
                ax.annotate("",
                            xy=(epoch_instance["coords"][real_index][0],
                                epoch_instance["coords"][real_index][1]),
                            xycoords='data',
                            xytext=(epoch_instance["coords"][previous_real_index][0],
                                    epoch_instance["coords"][previous_real_index][1]),
                            textcoords='data',
                            arrowprops=dict(arrowstyle="->",
                                            connectionstyle="arc3,rad=0.05",
                                            shrinkA=10, shrinkB=10, color=list_of_color[color_index%len(list_of_color)]),
                            )


    plt.savefig("sol_image.png")
    ax.clear()