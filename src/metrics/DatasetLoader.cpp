#include "DatasetLoader.hpp"

BoundingBox::BoundingBox(int l, Rect bb) {
    label = l;
    boundingBox = bb;
}

ostream& operator<<(ostream& os, const BoundingBox& bb) {
    Rect r = bb.boundingBox;
    os << "ID: " << bb.label << "; [" << r.x << ", " << r.y << ", " << r.width << ", " << r.height << "]";
    return os;
}

TrayData::TrayData(string p) {
    path = p;
    loadLabels();
    loadBoundingBoxes();
    loadMasks();
}

void TrayData::loadLabels() {
    // Generate the file path
    string filename = path + "/labels.txt";

    // Input file stream
    ifstream inFile;

    // Open the file
    inFile.open(filename);
    if (!inFile) {
        cerr << "Unable to open the file";
        exit(1);
    }

    // Extract the labels and store them in the foodLabels vector
    string line;
    int label;
    while( getline(inFile, line) ) {
        label = stoi(line);
        foodLabels.push_back(label);
    }

    inFile.close();
}

void TrayData::loadBoundingBoxes() {
    // Names of the files
    vector<string> filenames = {"food_image", "leftover1", "leftover2", "leftover3"};

    // Input file stream
    ifstream inFile;

    // For each filename, read the .txt, generate the Rects and store them in the vector
    for (int i=0; i<filenames.size(); i++) {
        // Path for the single file
        string filepath = path + "/bounding_boxes/" + filenames[i] + "_bounding_box.txt";

        inFile.open(filepath);
        if (!inFile) {
            cerr << "Unable to open the file: " << filepath;
            exit(1);
        }

        // Read each line
        string line;
        vector<BoundingBox> bb_singleFile;
        while( getline(inFile, line) ) {
            // Parse the line
            
            // Look for ';' and split the line accordingly
            string labelString = line.substr(0, line.find(";"));
            string rectString = line.substr(line.find(";")+3, line.length()-1);
            //cout << labelString << " + " << rectString << endl;

            // From labelString we extract the label => we cut the first 4 charachters
            int label = stoi( labelString.substr(4, labelString.length()) );
            //cout << label << endl;

            // From rectString we extract the coordinates
            vector<int> coordinates;
            int end = rectString.find(", ");
            while (end != -1) {
                int c = stoi( rectString.substr(0, end) );
                coordinates.push_back(c);
                rectString.erase(0, end+2);
                end = rectString.find(", ");
            }
            int c = stoi( rectString.substr(0, end) );
            coordinates.push_back(c);
            //for (int k=0; k<coordinates.size(); k++) { cout << coordinates[k] << " "; };

            // Build the rectangle object
            Rect rect(coordinates[0],coordinates[1],coordinates[2],coordinates[3]);

            // Build the BoundingBox object and store it in the array
            BoundingBox bb(label, rect);
            bb_singleFile.push_back(bb);
        }

        // Store the vector for a single file in the vector of the tray
        boundingBoxes.push_back(bb_singleFile);

        inFile.close();
    }
}

ostream& operator<<(ostream& os, const TrayData& tray) {
    os << "Base path: " << tray.path << endl;

    // Print the labels
    os << "\t" << "Labels: ";
    for (int i=0; i<tray.foodLabels.size(); i++) {
        os << tray.foodLabels[i] << ' ';
    }
    os << endl;

    // Print the bounding boxes
    os << "\t" << "Bounding boxes:" << endl;
    for (int i=0; i<tray.boundingBoxes.size(); i++) {
        os << "\t\tImage " << i << ":" << endl;
        for (int j=0; j<tray.boundingBoxes[i].size(); j++) {
            os << "\t\t\t" << tray.boundingBoxes[i][j] << endl;
        }
    }

    // Print something about the masks?

    return os;
}

void TrayData::loadMasks() {

}

void loadLabels_singleTray(string filename, vector<int>& labels) {
    // input stream variable
    ifstream inFile;

    // Open the file
    inFile.open(filename);
    if (!inFile) {
        cerr << "Unable to open the file";
        exit(1);
    }

    // Read the file
    // The labels are stored as numbers, one for each line 
    string line;
    int label;
    while( getline(inFile, line) ) {
        label = stoi(line);
        labels.push_back(label);
    }
}

void loadLabels(string foldername, vector<vector<int>>& labels) {
    // For each tray load the labels
    string filename;
    for (int i=1; i<=8; i++) {
        // Generate the filename corresponding to the tray
        filename = foldername + "/tray" + to_string(i) + "/labels.txt";
        //cout << filename << endl;

        // Load the labels in the output array
        loadLabels_singleTray(filename, labels[i-1]);
    }
}