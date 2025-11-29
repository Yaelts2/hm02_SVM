from datasets import (generate_linear_dataset, generate_Circular_dataset, plot_dataset)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from models import (split_train_test,
                    train_perceptron, 
                    train_linear_svm, 
                    train_kernel_svm,
                    train_poly_svm,
                    )
from evaluation import (plot_decision_boundary,
                        plot_accuracy_bars,
                        compute_confusion_and_f1,
                        plot_confusion_matrix,
                        )



def main():
    RUN_Q1_Q2 = True
    RUN_Q3 = True
    RUN_Q4 = True
    RUN_Q5 = True
    
    if RUN_Q1_Q2:
        ###  creating linear dataset
        X_lin, y_lin = generate_linear_dataset(n_samples_per_class=2048,random_state=0)
        #check if number of semple is correct after creating the data:
        print("Linear dataset shape:", X_lin.shape, y_lin.shape)
        plot_dataset(X_lin, y_lin, title="Linear dataset (2^12 samples)")
        
        ###  creating circular(non-linear) dataset
        X_nonlin, y_nonlin = generate_Circular_dataset(n_samples=4096, noise=0.2,random_state=0)
        #check if number of semple is correct after creating the data:
        print("Circular dataset shape:", X_nonlin.shape, y_nonlin.shape)
        plot_dataset(X_nonlin, y_nonlin, title="Circular dataset (moons, 2^12 samples)")
        
        
    if RUN_Q3 :
        ###Linear data
        #Split Train/Test
        X_train, X_test, y_train, y_test = split_train_test(X_lin, y_lin)

        #Model A: Perceptron 
        perceptron_model = train_perceptron(X_train, y_train)
        y_pred_perc = perceptron_model.predict(X_test)
        acc_perc = accuracy_score(y_test, y_pred_perc)

        #Model B: Linear SVM 
        linear_svm_model = train_linear_svm(X_train, y_train)
        y_pred_lin_svm = linear_svm_model.predict(X_test)
        acc_lin_svm = accuracy_score(y_test, y_pred_lin_svm)

        #Model C: Kernel SVM (RBF)
        kernel_svm_model = train_kernel_svm(X_train, y_train)
        y_pred_kernel_svm = kernel_svm_model.predict(X_test)
        acc_kernel_svm = accuracy_score(y_test, y_pred_kernel_svm)

        # Print results :
        print("\nClassification accuracy on the linear dataset:")
        print(f"  Perceptron accuracy:     {acc_perc:.4f}")
        print(f"  Linear SVM accuracy:     {acc_lin_svm:.4f}")
        print(f"  Kernel SVM (RBF) accuracy: {acc_kernel_svm:.4f}")

        ## Plot the results for linear model
        plot_decision_boundary(perceptron_model, X_lin, y_lin, title="Perceptron Decision Boundary (Linear dataset)")

        plot_decision_boundary(linear_svm_model, X_lin, y_lin, title="Linear SVM Decision Boundary (Linear dataset)")

        plot_decision_boundary(kernel_svm_model, X_lin, y_lin, title="Kernel SVM (RBF) Decision Boundary (Linear dataset)")

        #bar chart for the three models for the linear-data
        plot_accuracy_bars(
            accuracies=[acc_perc, acc_lin_svm, acc_kernel_svm],
            labels=["Perceptron", "Linear SVM", "Kernel SVM"],
            title="Accuracy Comparison – Linear Dataset"
        )


        ### non-linear data
        
        #Split Train/Test 
        X_train_nl, X_test_nl, y_train_nl, y_test_nl = split_train_test(X_nonlin, y_nonlin)

        #Model A: Perceptron (non-linear dataset)
        perceptron_model_nl = train_perceptron(X_train_nl, y_train_nl)
        y_pred_perc_nl = perceptron_model_nl.predict(X_test_nl)
        acc_perc_nl = accuracy_score(y_test_nl, y_pred_perc_nl)

        #Model B: Linear SVM (non-linear dataset) 
        linear_svm_model_nl = train_linear_svm(X_train_nl, y_train_nl)
        y_pred_lin_svm_nl = linear_svm_model_nl.predict(X_test_nl)
        acc_lin_svm_nl = accuracy_score(y_test_nl, y_pred_lin_svm_nl)

        #Model C: Kernel SVM (RBF) (non-linear dataset) 
        kernel_svm_model_nl = train_kernel_svm(X_train_nl, y_train_nl)
        y_pred_kernel_svm_nl = kernel_svm_model_nl.predict(X_test_nl)
        acc_kernel_svm_nl = accuracy_score(y_test_nl, y_pred_kernel_svm_nl)

        #Print results for the non-linear dataset
        print("\nClassification accuracy on the NON-LINEAR dataset (moons):")
        print(f"  Perceptron accuracy:       {acc_perc_nl:.4f}")
        print(f"  Linear SVM accuracy:       {acc_lin_svm_nl:.4f}")
        print(f"  Kernel SVM (RBF) accuracy: {acc_kernel_svm_nl:.4f}")

        #ploting the results for the non-linear data 
        plot_decision_boundary(perceptron_model_nl, X_nonlin, y_nonlin, title="Perceptron Decision Boundary (Non-linear dataset)")

        plot_decision_boundary(linear_svm_model_nl, X_nonlin, y_nonlin, title="Linear SVM Decision Boundary (Non-linear dataset)")

        plot_decision_boundary(kernel_svm_model_nl, X_nonlin, y_nonlin, title="Kernel SVM (RBF) Decision Boundary (Non-linear dataset)")

        #bar chart for the three models for the non-linear data
        plot_accuracy_bars(
            accuracies=[acc_perc_nl, acc_lin_svm_nl, acc_kernel_svm_nl],
            labels=["Perceptron", "Linear SVM", "Kernel SVM"],
            title="Accuracy Comparison – Non-linear Dataset"
        )

    if RUN_Q4:
        #Question4:
        #train the linear and non-linear data with the new kernel - Polynomial SVM
        poly_svm_model = train_poly_svm(X_train, y_train)
        y_pred_poly_svm = poly_svm_model.predict(X_test)
        acc_poly_svm = accuracy_score(y_test, y_pred_poly_svm)

        poly_svm_model_nl = train_poly_svm(X_train_nl, y_train_nl)
        y_pred_poly_svm_nl = poly_svm_model_nl.predict(X_test_nl)
        acc_poly_svm_nl = accuracy_score(y_test_nl, y_pred_poly_svm_nl)

        #print the accuracy:
        print("\nClassification accuracy on the linear dataset:")
        print(f"  Perceptron accuracy:         {acc_perc:.4f}")
        print(f"  Linear SVM accuracy:         {acc_lin_svm:.4f}")
        print(f"  Kernel SVM (RBF) accuracy:   {acc_kernel_svm:.4f}")
        print(f"  Poly SVM accuracy:           {acc_poly_svm:.4f}")
        print("\nClassification accuracy on the NON-LINEAR dataset (moons):")
        print(f"  Perceptron accuracy:         {acc_perc_nl:.4f}")
        print(f"  Linear SVM accuracy:         {acc_lin_svm_nl:.4f}")
        print(f"  Kernel SVM (RBF) accuracy:   {acc_kernel_svm_nl:.4f}")
        print(f"  Poly SVM accuracy:           {acc_poly_svm_nl:.4f}")


        ## confusion and f1 for linear data 
        cm_rbf_lin, f1_rbf_lin = compute_confusion_and_f1(kernel_svm_model, X_test, y_test)
        cm_poly_lin, f1_poly_lin = compute_confusion_and_f1(poly_svm_model, X_test, y_test)

        print("\n[Linear dataset] F1-scores:")
        print(f"  RBF SVM F1:   {f1_rbf_lin:.4f}")
        print(f"  Poly SVM F1:  {f1_poly_lin:.4f}")

        print("\n[Linear dataset] Confusion matrices (rows=true, cols=pred):")
        print("  RBF SVM:\n", cm_rbf_lin)
        print("  Poly SVM:\n", cm_poly_lin)

        ## confusion and f1 for non-linear data
        cm_rbf_nl, f1_rbf_nl = compute_confusion_and_f1(kernel_svm_model_nl, X_test_nl, y_test_nl)
        cm_poly_nl, f1_poly_nl = compute_confusion_and_f1(poly_svm_model_nl, X_test_nl, y_test_nl)

        print("\n[Non-linear dataset] F1-scores:")
        print(f"  RBF SVM F1:   {f1_rbf_nl:.4f}")
        print(f"  Poly SVM F1:  {f1_poly_nl:.4f}")

        print("\n[Non-linear dataset] Confusion matrices (rows=true, cols=pred):")
        print("  RBF SVM:\n", cm_rbf_nl)
        print("  Poly SVM:\n", cm_poly_nl)
        
        ### Ploting :   
        #Bar chart for F1 on linear dataset 
        plot_accuracy_bars(accuracies=[f1_rbf_lin, f1_poly_lin],
                        labels=["RBF SVM", "Poly SVM"],
                        title="F1-score Comparison – Linear Dataset",
                        ylabel="F1-score"
                        )
        #Bar chart for F1 on non-linear dataset 
        plot_accuracy_bars(accuracies=[f1_rbf_nl, f1_poly_nl],
                        labels=["RBF SVM", "Poly SVM"],
                        title="F1-score Comparison – Non-linear Dataset",
                        ylabel="F1-score"
                        )
        
        # Confusion matrices for linear data
        plot_confusion_matrix(cm_rbf_lin, ["Class 0", "Class 1"],
                            title="RBF SVM – Confusion Matrix (Linear)")
        plot_confusion_matrix(cm_poly_lin, ["Class 0", "Class 1"],
                            title="Poly SVM – Confusion Matrix (Linear)")

        # === Confusion matrices for non-linear data
        plot_confusion_matrix(cm_rbf_nl, ["Class 0", "Class 1"],
                            title="RBF SVM – Confusion Matrix (Non-linear)")
        plot_confusion_matrix(cm_poly_nl, ["Class 0", "Class 1"],
                            title="Poly SVM – Confusion Matrix (Non-linear)")

        plot_decision_boundary(poly_svm_model,X_lin, y_lin,title="Poly SVM Decision Boundary (Linear Dataset)")
        plot_decision_boundary(poly_svm_model_nl, X_nonlin, y_nonlin,title="Poly SVM Decision Boundary (Non-linear Dataset)")

    
    if RUN_Q5:
        # Question 5 a
        k = 150  # number of repetitions
        low_sd = 0.1
        high_sd = 0.4

        f1_low_list = []
        f1_high_list = []

        for i in range(k):
            # sample 1000 points
            idx = np.random.choice(len(X_nonlin), size=1000, replace=False)
            X_base = X_nonlin[idx]
            y_base = y_nonlin[idx]

            # low noise
            noise_low = np.random.normal(0, low_sd, X_base.shape)
            X_low = X_base + noise_low
            _, f1_low = compute_confusion_and_f1(kernel_svm_model_nl, X_low, y_base)
            f1_low_list.append(f1_low)

            # high noise
            noise_high = np.random.normal(0, high_sd, X_base.shape)
            X_high = X_base + noise_high
            _, f1_high = compute_confusion_and_f1(kernel_svm_model_nl, X_high, y_base)
            f1_high_list.append(f1_high)

        # aggregate
        f1_low_mean = np.mean(f1_low_list)
        f1_high_mean = np.mean(f1_high_list)

        print(f"\nLow noise (SD={low_sd}):  mean F1 = {f1_low_mean:.4f}")
        print(f"High noise (SD={high_sd}): mean F1 = {f1_high_mean:.4f}")

        # plot
        plot_accuracy_bars(accuracies=[f1_low_mean, f1_high_mean],
                        labels=["Low noise", "High noise"],
                        title="RBF SVM – robustness to input noise",
                        ylabel="Mean F1-score",
                        )

        # sample one batch for visualization 
        idx = np.random.choice(len(X_nonlin), size=1000, replace=False)
        X_base = X_nonlin[idx]
        y_base = y_nonlin[idx]

        # create low-noise and high-noise versions for visualization
        X_low_vis = X_base + np.random.normal(0, low_sd, X_base.shape)
        X_high_vis = X_base + np.random.normal(0, high_sd, X_base.shape)

        # plot decision boundary for low noise
        plot_decision_boundary(kernel_svm_model_nl,
                            X_low_vis,
                            y_base,
                            title="RBF SVM Decision Boundary (Low noise)",
                            )

        # plot decision boundary for high noise
        plot_decision_boundary(kernel_svm_model_nl,
                            X_high_vis,
                            y_base,
                            title="RBF SVM Decision Boundary (High noise)",
                            )
                
        # Histograms of F1 distributions 
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(f1_low_list, bins=15, color="red", alpha=0.7)
        plt.title("F1 distribution – Low noise")
        plt.xlabel("F1-score")
        plt.ylabel("Count")
        plt.subplot(1, 2, 2)
        plt.hist(f1_high_list, bins=15, color="blue", alpha=0.7)
        plt.title("F1 distribution – High noise")
        plt.xlabel("F1-score")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()


        #question 5 b
        label_noise_levels = [0.0, 0.15, 0.30, 0.50]
        k_label = 50 
        f1_means_label = []

        for noise_level in label_noise_levels:
            f1_list = []
            for i in range(k_label):
                y_train_noisy = y_train_nl.copy()
                n_corrupt = int(noise_level * len(y_train_noisy))

                if n_corrupt > 0:
                    idx_corrupt = np.random.choice(len(y_train_noisy), size=n_corrupt, replace=False)

                    # flip binary labels 0 <-> 1
                    y_train_noisy[idx_corrupt] = 1 - y_train_noisy[idx_corrupt]

                # train RBF SVM on the corrupted labels
                svm_noisy = train_kernel_svm(X_train_nl, y_train_noisy)
                _, f1 = compute_confusion_and_f1(svm_noisy, X_test_nl, y_test_nl)
                f1_list.append(f1)

            mean_f1 = np.mean(f1_list)
            f1_means_label.append(mean_f1)
            print(f"Label noise {int(noise_level*100)}%: mean F1 = {mean_f1:.4f}")

        # Plot performance vs. percentage of shuffled labels
        percentages = [int(p*100) for p in label_noise_levels]

        plot_accuracy_bars(
            accuracies=f1_means_label,
            labels=[f"{p}%" for p in percentages],
            title="RBF SVM – performance vs. label noise (Circular dataset)",
            ylabel="Mean F1-score"
        )
            
main()