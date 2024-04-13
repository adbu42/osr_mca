import matplotlib.pyplot as plt
import pandas

data = pandas.read_csv('condition_difference.csv', delimiter=',')
# Extract relevant columns
training_step = data['Step']
error_rate_model1 = data['glorious-field-1 - val_condition_difference']
error_rate_model2 = data['electric-rain-3 - val_condition_difference']
# Create a line plot
plt.plot(training_step, error_rate_model1, label='original U-Net (Ronneberger et al.)')
plt.plot(training_step, error_rate_model2, label='adapted U-Net')

# Customize the plot
plt.xlabel('Training Step')
plt.ylabel('condition difference')
plt.title('Validation condition difference')
plt.legend()

# Show the plot
plt.show()
