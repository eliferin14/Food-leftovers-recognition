#include <iostream>
#include "MeanAveragePrecision.hpp"
#include "DatasetLoader.hpp"
#include "IntersectionOverUnion.hpp"
#include "../utils.hpp"

using namespace std;
using namespace cv;

/*void loadLabels_test(string filename) {
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
}*/

int main(int argc, char** argv) {
    string ourDatasetPath = "../Our_dataset";
    string trueDatasetPath = "../Food_leftover_dataset";

    std::vector<std::string> filenames = {"food_image", "leftover1", "leftover2", "leftover3"};

    // BOUNDING BOXES METRICS

    vector<vector<Rect>> bb_tray(4);
    vector<vector<vector<Rect>>> ourBoundingBoxes(8, bb_tray);
    vector<vector<vector<Rect>>> trueBoundingBoxes(8, bb_tray);

    loadBoundingBoxes(ourDatasetPath, ourBoundingBoxes);
    loadBoundingBoxes(trueDatasetPath, trueBoundingBoxes);

    // Mean iou of the four pictures, considering each tray separately
    vector<double> meanIouPerTrayBB;
    cout << "\nMean IoU per tray (bb): " << endl;
    for (int tray=1; tray<=8; tray++) {

        double iou = 0;

        for (int img=0; img<4; img++) {
            iou += iou_twoImagesUnionBB(ourBoundingBoxes[tray-1][img], trueBoundingBoxes[tray-1][img]);
        }

        iou /= 4;

        meanIouPerTrayBB.push_back(iou);

        cout << "\tTray " << tray << ": " << iou << endl;
    }

    // Mean iou of the trays, considering each image separately
    vector<double> meanIouPerImageBB;
    cout << "\nMean IoU per image (bb): " << endl;
    for (int img=0; img<4; img++) {

        double iou = 0;

        for (int tray=1; tray<=8; tray++) {
            iou += iou_twoImagesUnionBB(ourBoundingBoxes[tray-1][img], trueBoundingBoxes[tray-1][img]);
        }

        iou /= 8;

        meanIouPerImageBB.push_back(iou);

        cout << "\t" << filenames[img] << ": " << iou << endl;
    }



    // MASKS METRICS
    vector<Mat> mask_tray(4);
    vector<vector<Mat>> ourMasks(8, mask_tray);
    vector<vector<Mat>> trueMasks(8, mask_tray);

    loadMasks(ourDatasetPath, ourMasks, false);
    loadMasks(trueDatasetPath, trueMasks, true);

    int a = 4;
    int b = 2;
    cout << iou_twoMasks(ourMasks[a][b], trueMasks[a][b]) << endl;

    // Mean iou of the four pictures, considering each tray separately
    vector<double> meanIouPerTrayMasks;
    cout << "\nMean IoU per tray (masks): " << endl;
    for (int tray=1; tray<=8; tray++) {

        double iou = 0;

        for (int img=0; img<4; img++) {
            iou += iou_twoMasks(ourMasks[tray-1][img], trueMasks[tray-1][img]);
            //printf("%f\n", iou);
        }

        iou /= 4;

        meanIouPerTrayMasks.push_back(iou);

        cout << "\tTray " << tray << ": " << iou << endl;
    }

    // Mean iou of the trays, considering each image separately
    vector<double> meanIouPerImageMasks;
    cout << "\nMean IoU per image (masks): " << endl;
    for (int img=0; img<4; img++) {

        double iou = 0;

        for (int tray=1; tray<=8; tray++) {
            iou += iou_twoMasks(ourMasks[tray-1][img], trueMasks[tray-1][img]);
            //printf("%f\n", iou);
        }

        iou /= 8;

        meanIouPerImageMasks.push_back(iou);

        cout << "\t" << filenames[img] << ": " << iou << endl;
    }

}