import matplotlib.pyplot as plt
import random
# Create a list of x values
# loss_x = [list(range(10)), list(range(10)), list(range(10))]

# # Create a list of y values corresponding to x
# loss_y = []


# for i in range(len(loss_x)):
#     y = []
#     for j in range(10):
#         y.append(random.randrange(1, 100))
#     loss_y.append(y)


# plt.figure()
# # Loop through each set of x and y values
# for i in range(len(loss_x)):
#     # Create a new figure
    
#     # Plot the x and y values
#     plt.plot(loss_x[i], loss_y[i])
    
#     # Set labels and title
#     plt.xlabel('X Axis Label')
#     plt.ylabel('Y Axis Label')
#     plt.title(f'Plot {i+1}')
    
#     # Save the figure as a png file
#     plt.savefig(f'plot_{i+1}.png')  
    
#     # Clear the figure for the next plot
# plt.clf()

# # Close all figure windows
# plt.close('all')

import datetime

now = datetime.datetime.now()
current_time = now.strftime("%m/%d/%H:%M")

print("Current Time =", current_time)