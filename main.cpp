
// Usage example:  ./object_detection_yolo.out --video=run.mp4
//                 ./object_detection_yolo.out --image=bird.jpg
#include <fstream>
#include <sstream>
#include <iostream>
#include <memory>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/opencv.hpp>



const char* keys =
"{help h usage ? | | Usage examples: \n\t\t./PEOPLE_TRACKER --video=video.mp4}"
"{skip_n n        |<none>| N frames object detector   }"
"{video v       |<none>| input video   }"
"{m_cfg m       |<none>| model configuration}"
"{m_weights w   |<none>| model weights}"

;
using namespace cv;
using namespace dnn;
using namespace std;

// Initialize the parameters
float confThreshold = 0.9; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image
vector<string> classes;

// Fill the vector with random colors
void getRandomColors(vector<Scalar>& colors, int numColors)
{
  RNG rng(0);
  for(int i=0; i < numColors; i++)
    colors.push_back(Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255))); 
}



// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs, vector<Rect>& boxes);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net);

vector<string> trackerTypes = {"BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"}; 

// create tracker by name
Ptr<Tracker> createTrackerByName(string trackerType) 
{
  Ptr<Tracker> tracker;
  if (trackerType ==  trackerTypes[0])
    tracker = TrackerBoosting::create();
  else if (trackerType == trackerTypes[1])
    tracker = TrackerMIL::create();
  else if (trackerType == trackerTypes[2])
    tracker = TrackerKCF::create();
  else if (trackerType == trackerTypes[3])
    tracker = TrackerTLD::create();
  else if (trackerType == trackerTypes[4])
    tracker = TrackerMedianFlow::create();
  else if (trackerType == trackerTypes[5])
    tracker = TrackerGOTURN::create();
  else if (trackerType == trackerTypes[6])
    tracker = TrackerMOSSE::create();
  else if (trackerType == trackerTypes[7])
    tracker = TrackerCSRT::create();
  else {
    cout << "Incorrect tracker name" << endl;
    cout << "Available trackers are: " << endl;
    for (vector<string>::iterator it = trackerTypes.begin() ; it != trackerTypes.end(); ++it)
      std::cout << " " << *it << endl;
  }
  return tracker;
}


int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run object detection using YOLO3 in OpenCV.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    // Load names of classes
    string classesFile = "coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);
    
    // Give the configuration and weight files for the model
    String modelConfiguration = "yolov3.cfg";
    String modelWeights = "yolov3.weights";

    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
    
    cout << "network uploaded..." << endl;
    // Open a video file or an image file or a camera stream.
    string str, outputFile;
    VideoCapture cap;
    VideoWriter video;
    Mat frame, blob;
    
    try {
        
        outputFile = "tracker_out.avi";
      
        // Open the video file
        str = parser.get<String>("video");
        ifstream ifile(str);
        if (!ifile) throw("error");
        cap.open(str);
        str.replace(str.end()-4, str.end(), "tracker_out.avi");
        outputFile = str;
        cout << "Reading file..." << endl;

        
    }
    catch(...) {
        cout << "Could not open the input image/video stream" << endl;
        return 0;
    }
    
    // Get the video writer initialized to save the output video
    
    video.open(outputFile, VideoWriter::fourcc('M','J','P','G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
   
    // tracker
    // Multitracker
    // Specify the tracker type
  
    string trackerType = "CSRT";
    // Create multitracker
    Ptr<MultiTracker> multiTracker = cv::MultiTracker::create();
    // Pointer
    Ptr<MultiTracker>* Pointer = &multiTracker;
    

    // First frame
    cap >> frame;
    blobFromImage(frame, blob, 1/255.0, cvSize(inpWidth, inpHeight), Scalar(0,0,0), true, false);
    net.setInput(blob);
    vector<Mat> outs;
    net.forward(outs, getOutputsNames(net));
    vector<Rect> boxes;
    postprocess(frame, outs, boxes);
    int total_frames = 0;
    vector<Scalar> colors;  
    getRandomColors(colors, boxes.size()); 
    
    // Initialize multitracker
    for(int i=0; i<boxes.size(); i++)
        multiTracker->add(createTrackerByName(trackerType), frame, Rect2d(boxes[i]));  


    // Create a window
    static const string kWinName = "Tracking People";
    namedWindow(kWinName, WINDOW_NORMAL);
    imshow(kWinName, frame);
    // Process frames.
    while (waitKey(1) < 0)
    {
        if (total_frames % 40 == 0)
        {
        
            
            // get frame from the video
            cap >> frame;
            total_frames++;
            blobFromImage(frame, blob, 1/255.0, cvSize(inpWidth, inpHeight), Scalar(0,0,0), true, false);
            net.setInput(blob);
            vector<Mat> outs;
            net.forward(outs, getOutputsNames(net));
            vector<Rect> boxes;

            postprocess(frame, outs, boxes);

            vector<Scalar> colors;  
            getRandomColors(colors, boxes.size()); 


            // Initialize again multitracker
            *Pointer = cv::MultiTracker::create();
			
			
            //Ptr<MultiTracker> multiTracker = cv::MultiTracker::create();

            
            for(int i=0; i<boxes.size(); i++)
                multiTracker->add(createTrackerByName(trackerType), frame, Rect2d(boxes[i]));
                  
            


            Mat detectedFrame;
            frame.convertTo(detectedFrame, CV_8U);

            if (parser.has("skip_n")) imwrite(outputFile, detectedFrame);
            else video.write(detectedFrame);
            
            imshow(kWinName, frame);

        }

        else {
            // Stop the program if reached end of video
            if (frame.empty()) {
                cout << "Done processing !!!" << endl;
                cout << "Output file is stored as " << outputFile << endl;
                waitKey(3000);
                break;
            }


            cap >> frame;
            total_frames++;

            


            //Update tracker.
            
            multiTracker->update(frame);
            

                       
              

            // Draw tracked objects          
            
            for(unsigned i=0; i<multiTracker->getObjects().size(); i++)
            {
                
                rectangle(frame, multiTracker->getObjects()[i], Scalar( 255, 0, 0 ), 2, 1);
            }

            //rectangle(frame, multiTracker->getObjects().back(), Scalar( 255, 0, 0 ), 2, 1);

            
            //Sets the input to the network

            
            // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)

            string label = " CQS People Tracker";
            putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
            
            // Write the frame with the detection boxes
            Mat detectedFrame;
            frame.convertTo(detectedFrame, CV_8U);
            if (parser.has("skip_n")) imwrite(outputFile, detectedFrame);
            else video.write(detectedFrame);
            
            imshow(kWinName, frame);
        }
        
    }
    
    cap.release();
    if (!parser.has("skip_n")) video.release();

    return 0;
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs, vector<Rect>& boxes)
{
    vector<int> classIds;
    vector<float> confidences;
    
    
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if ((confidence > confThreshold) && (classIdPoint.x == 0))
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }

    
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}