import HelperMethods

saved = HelperMethods.retrieve_history()
losses = []
accuraces = []
val_losses = []
val_accuraces = []
epoch_times = []
params = []
for save in saved:
  losses.append(saved[save]['loss'])
  accuraces.append(saved[save]['accuracy'])
  val_accuraces.append(saved[save]['val_accuracy'])
  val_losses.append(saved[save]['val_loss'])
  epoch_times.append(saved[save]['epoch_time'])
  params.append(saved[save]['param'])

  print(str(saved[save]['param'])+" train accuracy: "+str(saved[save]['train_accuracy'])+"%")
  print(str(saved[save]['param']) + " test accuracy: " + str(saved[save]['test_accuracy'])+"%")
  print(str(saved[save]['param']) + " time: " + str(saved[save]['time']))
  print("------------------------------------------")

HelperMethods.plot_result_multiple(losses,"Loss Comparison","Loss","Epoch",params)
HelperMethods.plot_result_multiple(accuraces,"Accuracy Comparison","Accuracy","Epoch",params)
HelperMethods.plot_result_multiple(val_accuraces,"Val Accuracy Comparison","Accuracy","Epoch",params)
HelperMethods.plot_result_multiple(val_losses,"Val Loss Comparison","Loss","Epoch",params)
HelperMethods.plot_result_multiple(epoch_times,"Epoch Time Comparison","Time","Epoch",params)
