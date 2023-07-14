#include <iostream>
#include "MeanAveragePrecision.hpp"
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

void getPrecisionRecall_test() {
    Rect rect1(100, 100, 200, 200);
    Rect rect2(200, 200, 300, 300);
    BoundingBox bb1 = BoundingBox(1, rect1);
    BoundingBox bb2 = BoundingBox(1, rect2);

    double precision;
    double recall;
    getPrecisionRecall(bb1, bb2, precision, recall);
}

int main(int argc, char** argv) {
    getPrecisionRecall_test();
}