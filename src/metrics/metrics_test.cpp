#include <iostream>
#include "DatasetLoader.hpp"

void loadLabels_test(string filename) {
    vector<vector<int>> labels(8);
    loadLabels(filename, labels);
    for (int i=0; i<labels.size(); i++) {
        printf("Tray %d: [%d] < ", i+1, (int)labels[i].size());
        for (int j=0; j<labels[i].size(); j++) {
            printf("%d ", labels[i][j]);
        }
        printf(">\n");
    }
}

void TrayData_labels_test(string path) {
    TrayData tray(path);

    for (int j=0; j<tray.foodLabels.size(); j++) {
        printf("%d ", tray.foodLabels[j]);
    }
    printf("\n");
}

void TrayData_bb_test(string path) {
    TrayData tray(path);

    cout << tray;
}

int main(int argc, char** argv) {
    string path = argv[1];
    TrayData_bb_test(path);
}