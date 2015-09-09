#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include "Dataman.h"
#include "bpnet.h"

using namespace std;
using namespace cv;

#define PATTERN_COUNT 30
#define PATTERN_SIZE 625
#define NETWORK_INPUTNEURONS 626
#define NETWORK_OUTPUT 3
#define HIDDEN_LAYERS 1

Mat detectAndDisplay( Mat frame );

String face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;

int main(int ac, char** av)
{
	double t = (double)getTickCount();
	
    int hl[] = {10};
	int train;
	if (ac > 1)
	{
		train = 1;
		cout << "training" << endl;
	}
	
	Dataman* data = new Dataman((int)NETWORK_INPUTNEURONS,(int)NETWORK_OUTPUT,hl[0]);

    float pattern[PATTERN_COUNT][PATTERN_SIZE];
    float desiredout[PATTERN_COUNT][NETWORK_OUTPUT];
    
	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
	
	Mat frame, dface;
	
	bpnet net;
    int i,j;
    float error;
    net.create(PATTERN_SIZE,NETWORK_INPUTNEURONS,NETWORK_OUTPUT,hl,HIDDEN_LAYERS);
		
	if (train == 1)
	{
		frame = imread("faces/s4/1.pgm", CV_LOAD_IMAGE_COLOR);

        dface = detectAndDisplay( frame );
        dface = dface > 128;
		uchar* p;
		for(int ia = 0; ia < dface.rows; ++ia)
		{
			p = dface.ptr<uchar>(ia);
			for (int ja = 0; ja < dface.cols*dface.channels(); ++ja)
			{
				pattern[PATTERN_COUNT-1][ja+ia*25] = (int)p[ja]/255.0;
				//~ for (int k = 0; k < NETWORK_OUTPUT; ++k)
				//~ {
					//~ desiredout[PATTERN_COUNT-1][k] = 0;
				//~ }
				//~ desiredout[PATTERN_COUNT-1][PATTERN_COUNT-1] = 1;
				desiredout[PATTERN_COUNT-1][0] = 1;
				desiredout[PATTERN_COUNT-1][1] = 1;
			}
		}       
		
		for (int iz = 0; iz < PATTERN_COUNT-1; iz++)
		{			
			cout << "load gambar dr database " << iz+1 << endl;
			char tfoto[32];
			sprintf(tfoto,"faces/s4/1.pgm",iz+1);
			frame = imread(tfoto, CV_LOAD_IMAGE_COLOR);
			
			if( frame.empty() )
			{
				printf(" --(!) No captured frame -- Break!");
				break;
			}

			//-- 3. Apply the classifier to the frame
			dface = detectAndDisplay( frame );
			
			dface = dface > 128;
			uchar* p;
			for(int ia = 0; ia < dface.rows; ++ia)
			{
				p = dface.ptr<uchar>(ia);
				for (int ja = 0; ja < dface.cols*dface.channels(); ++ja)
				{
					pattern[iz][ja+ia*25] = (int)p[ja]/255.0;
				}
			}
			
			desiredout[iz][0] = 0;
			desiredout[iz][1] = 0;
			//~ desiredout[iz][2] = 0;
			//~ desiredout[iz][3] = 0;
			
			int ks = iz;
			if (ks > 8)	
			{
				desiredout[iz][3] = 1;
				ks = ks -8;
			}
			if (ks > 4)	
			{
				desiredout[iz][2] = 1;
				ks = ks -4;
			}
			if (ks > 2)	
			{
				desiredout[iz][1] = 1;
				ks = ks-2;
			}
			desiredout[iz][0] = ks;
			

			int c = waitKey(10);
			if( (char)c == 27 ) { break; } // escape
			
		}
		cout << "start training " << endl;
		
		error = 1;
		int isd = 0;
		while(error > 0.001)
		{
			isd++;
			cout << "ITER: " << isd << " " << error << endl;
			error=0;
			for(j=0;j<PATTERN_COUNT;j++)
			{
				error+=net.train(desiredout[j],pattern[j],0.2f,0.1f);
			}
			error/=PATTERN_COUNT;
			
			
			int c = waitKey(10);
			if( (char)c == 27 ) { break; } // escape
		}
		
		for(i=0;i<net.m_outputlayer.neuroncount;i++)
		{
			for(j=0;j<net.m_outputlayer.inputcount;j++)
			{
				data->out_weight[i][j] = net.m_outputlayer.neurons[i]->weights[j];
			}
			data->out_wgain[i] = net.m_outputlayer.neurons[i]->wgain;
		}

		for(i=0;i<net.m_hiddenlayers[HIDDEN_LAYERS-1]->neuroncount;i++)
		{
			for(j=0;j<net.m_hiddenlayers[HIDDEN_LAYERS-1]->inputcount;j++)
			{
				data->hid_weight[i][j] = net.m_hiddenlayers[HIDDEN_LAYERS-1]->neurons[i]->weights[j];
			}
			data->hid_wgain[i] = net.m_hiddenlayers[HIDDEN_LAYERS-1]->neurons[i]->wgain;
		}

		for(i=0;i<net.m_inputlayer.neuroncount;i++)
		{
			for(j=0;j<net.m_inputlayer.inputcount;j++)
			{
				data->in_weight[i][j] = net.m_inputlayer.neurons[i]->weights[j];
			}
			data->in_wgain[i] = net.m_inputlayer.neurons[i]->wgain;
		}
		data->write();
	}
	else
	{
		data->read();
		for(i=0;i<net.m_outputlayer.neuroncount;i++)
		{
			for(j=0;j<net.m_outputlayer.inputcount;j++)
			{
				net.m_outputlayer.neurons[i]->weights[j]= data->out_weight[i][j];
			}
			net.m_outputlayer.neurons[i]->wgain = data->out_wgain[i];
		}
		
		for(i=0;i<net.m_hiddenlayers[HIDDEN_LAYERS-1]->neuroncount;i++)
		{
			for(j=0;j<net.m_hiddenlayers[HIDDEN_LAYERS-1]->inputcount;j++)
			{
				net.m_hiddenlayers[HIDDEN_LAYERS-1]->neurons[i]->weights[j]= data->hid_weight[i][j];
			}
			net.m_hiddenlayers[HIDDEN_LAYERS-1]->neurons[i]->wgain = data->hid_wgain[i];
		}

		for(i=0;i<net.m_inputlayer.neuroncount;i++)
		{
			for(j=0;j<net.m_inputlayer.inputcount;j++)
			{
				net.m_inputlayer.neurons[i]->weights[j]= data->in_weight[i][j];	
			}
			net.m_inputlayer.neurons[i]->wgain = data->in_wgain[i];
		}
		
		for (int iz = 0; iz < PATTERN_COUNT-20; iz++)
		{
			char tfoto[32];
			sprintf(tfoto,"faces/s1/%d.pgm",iz+1);
			frame = imread(tfoto, CV_LOAD_IMAGE_COLOR);
			
			if( frame.empty() )
			{
				printf(" --(!) No captured frame -- Break!\n");
				break;
			}

			//-- 3. Apply the classifier to the frame
			dface = detectAndDisplay( frame );
			
			dface = dface > 128;
			uchar* p;
			int ss = 0;
			for(int ia = 0; ia < dface.rows; ++ia)
			{
				p = dface.ptr<uchar>(ia);
				for (int ja = 0; ja < dface.cols*dface.channels(); ++ja)
				{
					if (ja > 26 and ss == 0) 
					{
						cout << "false detect: " << tfoto << endl;
						ss = 1;
					}
					pattern[iz][ja+ia*25] = (int)p[ja]/255.0;
				}
			}
		}
			
		for (int iz = 0; iz < PATTERN_COUNT-20; iz++)
		{
			char tfoto[32];
			sprintf(tfoto,"faces/s5/%d.pgm",iz+1);
			frame = imread(tfoto, CV_LOAD_IMAGE_COLOR);
			
			if( frame.empty() )
			{
				printf(" --(!) No captured frame -- Break!\n");
				break;
			}

			//-- 3. Apply the classifier to the frame
			dface = detectAndDisplay( frame );
			
			dface = dface > 128;
			uchar* p;
			int ss = 0;
			for(int ia = 0; ia < dface.rows; ++ia)
			{
				p = dface.ptr<uchar>(ia);
				for (int ja = 0; ja < dface.cols*dface.channels(); ++ja)
				{
					if (ja > 26 and ss == 0) 
					{
						cout << "false detect: " << tfoto << endl;
						ss = 1;
					}
					pattern[iz+10][ja+ia*25] = (int)p[ja]/255.0;
				}
			}

			int c = waitKey(10);
			if( (char)c == 27 ) { break; } // escape
			
		}
		
		for (int iz = 0; iz < PATTERN_COUNT-20; iz++)
		{
			char tfoto[32];
			sprintf(tfoto,"faces/s6/%d.pgm",iz+1);
			frame = imread(tfoto, CV_LOAD_IMAGE_COLOR);
			
			if( frame.empty() )
			{
				printf(" --(!) No captured frame -- Break!\n");
				break;
			}

			//-- 3. Apply the classifier to the frame
			dface = detectAndDisplay( frame );
			
			dface = dface > 128;
			uchar* p;
			int ss = 0;
			for(int ia = 0; ia < dface.rows; ++ia)
			{
				p = dface.ptr<uchar>(ia);
				for (int ja = 0; ja < dface.cols*dface.channels(); ++ja)
				{
					if (ja > 26 and ss == 0) 
					{
						cout << "false detect: " << tfoto << endl;
						ss = 1;
					}
					pattern[iz+20][ja+ia*25] = (int)p[ja]/255.0;
				}
			}

			int c = waitKey(10);
			if( (char)c == 27 ) { break; } // escape
			
		}
	}
	

    for(i=0;i<PATTERN_COUNT;i++)
    {
        net.propagate(pattern[i]);
    //display result
        cout << "TESTED PATTERN " << i+1 << " NET RESULT: "<< " " << net.getOutput().neurons[0]->output << " " << net.getOutput().neurons[1]->output << endl;
        if ((i+1)%10 == 0) cout << endl;
        //cout << "at ITER:" << isd << endl;
    }

	
	t = ((double)getTickCount() - t)/getTickFrequency();
	cout << "Times passed in seconds: " << t << endl;
    return 0;
}

Mat detectAndDisplay( Mat frame )
{
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    Mat ff(25, 25, CV_8UC3);

    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        Mat faceROI = frame_gray( faces[i] );
        resize(faceROI, ff, ff.size(), 0, 0, INTER_CUBIC);
        //imshow( "face_es", ff );
    }
    //-- Show what you got
    //imshow( window_name, frame );
    return ff;
}
