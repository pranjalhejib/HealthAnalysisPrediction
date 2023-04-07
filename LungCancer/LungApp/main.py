from training import trainingProcess, buildNetwork, buildVgg
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cancer1 = 'D:\\HealthCareNew-PRMITR\\LungCancer\\LungApp\\data\\cancer'
healthy1 = 'D:\\HealthCareNew-PRMITR\\LungCancer\\LungApp\\data\\healthy'

# training image paths
def init():
    X_train, X_test, Y_train, Y_test = trainingProcess(cancer1,healthy1)
    print('Training X shape =',X_train.shape)
    print('Training y shape =',Y_train.shape)
    print('Testing X shape =',X_test.shape)
    print('Testing y shape =',Y_test.shape)
    model = buildNetwork()
    print(model.summary())
    epochs = 5
    model.fit(X_train, Y_train, epochs=epochs) # training phase
    Y_pred=model.predict(X_test)
    print("Y Pred :: ")
    print(Y_pred)
    print("Y Test :: ")
    print(Y_test)

    Y_pred=np.argmax(Y_pred, axis=1)
    Y_test=np.argmax(Y_test, axis=1)
    cm = confusion_matrix(Y_pred,Y_test)
    ac = accuracy_score(Y_pred,Y_test)
    cr = classification_report(Y_pred,Y_test)
    print(cm)
    print(ac)
    print(cr)
    #model.save('NN.h5')
    print("Model_Ready")
    #testingProcess()
    #cm_nor = cm.mean()/cm.std()
    print(cm)   
    ax = sns.heatmap(cm/np.sum(cm), annot=True,  fmt='.2%', cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    plt.savefig(r"d:\cf.png")
    plt.show()
    
def vgg():
    model = buildVgg()
    print(model.summary())
    epochs = 1
    model.fit(X_train, Y_train, epochs=epochs) # training phase
    Y_pred=model.predict(X_test)
    print(Y_pred)
    Y_pred=np.argmax(Y_pred, axis=1)
    Y_test=np.argmax(Y_test, axis=1)
    cm = confusion_matrix(Y_pred,Y_test)
    ac = accuracy_score(Y_pred,Y_test)
    cr = classification_report(Y_pred,Y_test)
    print(cm)
    print(ac)
    print(cr)
    #model.save('NN.h5')
    print("Model_Ready")
    #testingProcess()
    #cm_nor = cm.mean()/cm.std()
    print(cm)
    ax = sns.heatmap(cm/np.sum(cm), annot=True,  fmt='.2%', cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    plt.savefig(r"d:\cfvgg.png")
    plt.show()

init()