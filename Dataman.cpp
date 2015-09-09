#include <opencv2/core/core.hpp>
#include <iostream>

#include "Dataman.h"

using namespace std;
using namespace cv;

Dataman::Dataman(int nneuron_input,int nneuron_output, int nneuron_hidden)
{
	ineur = nneuron_input;
	oneur = nneuron_output;
	hneur = nneuron_hidden;
	
	in_wgain.resize(nneuron_input);
	in_weight.resize(nneuron_input);
	for (int i = 0; i < nneuron_input; ++i)
	{
		in_weight[i].resize(nneuron_input-1);
	}
	
	hid_wgain.resize(nneuron_hidden);
	hid_weight.resize(nneuron_hidden);
	for (int i = 0; i < nneuron_hidden; ++i)
	{
		hid_weight[i].resize(nneuron_input);
	}
	
	out_wgain.resize(nneuron_output);
	out_weight.resize(nneuron_output);
	for (int i = 0; i < nneuron_output; ++i)
	{
		out_weight[i].resize(nneuron_hidden);
	}
}

Dataman::~Dataman()
{
}

void Dataman::write()
{
	fs.open("data.xml", FileStorage::WRITE);
	if (!fs.isOpened())
	{
		cout << "Failed to open " << endl;
	}
	fs << "out_weight" << "[";
	for (int i = 0; i < oneur; ++i)
	{
		for (int j = 0; j < hneur; ++j)
		{
			fs << out_weight[i][j];
		}
	}
	fs << "]";
	
	fs << "out_wgain" << "[";
	for (int i = 0; i < oneur; ++i)
	{
		fs << out_wgain[i];
	}
	fs << "]";
	
	
	fs << "hid_weight" << "[";
	for (int i = 0; i < hneur; ++i)
	{
		for (int j = 0; j < ineur; ++j)
		{
			fs << hid_weight[i][j];
		}
	}
	fs << "]";
	
	fs << "hid_wgain" << "[";
	for (int i = 0; i < hneur; ++i)
	{
		fs << hid_wgain[i];
	}
	fs << "]";
	
	fs << "in_weight" << "[";
	for (int i = 0; i < ineur; ++i)
	{
		for (int j = 0; j < ineur - 1; ++j)
		{
			fs << in_weight[i][j];
		}
	}
	fs << "]";
	
	fs << "in_wgain" << "[";
	for (int i = 0; i < ineur; ++i)
	{
		fs << in_wgain[i];
	}
	fs << "]";
	
	fs.release();
}

void Dataman::read()
{
	fs.open("data.xml", FileStorage::READ);
	
	if (!fs.isOpened())
	{
		cout << "Failed to open " << endl;
	}
	
	FileNode n;
	FileNodeIterator it;
	FileNodeIterator it_end;
	int i,j;
	
	i = 0;
	n = fs["out_weight"];
	it = n.begin();
	it_end = n.end();
	j = 0;
	for (; it != it_end; ++it)
	{
		out_weight[i][j] = (float)*it;
		j++;
		if (j == hneur) 
		{
			j = 0;
			i++;
		}
	}
		
	i = 0;
	n = fs["out_wgain"];
	it = n.begin();
	it_end = n.end();
	for (; it != it_end; ++it)
	{
		out_wgain[i] = (float)*it;
		i++;
	}
	
	i = 0;
	n = fs["hid_weight"];
	it = n.begin();
	it_end = n.end();
	j = 0;
	for (; it != it_end; ++it)
	{
		hid_weight[i][j] = (float)*it;
		j++;
		if (j == ineur) 
		{
			j = 0;
			i++;
		}
	}
		
	i = 0;
	n = fs["hid_wgain"];
	it = n.begin();
	it_end = n.end();
	for (; it != it_end; ++it)
	{
		hid_wgain[i] = (float)*it;
		i++;
	}
	
	i = 0;
	n = fs["in_weight"];
	it = n.begin();
	it_end = n.end();
	j = 0;
	for (; it != it_end; ++it)
	{
		in_weight[i][j] = (float)*it;
		j++;
		if (j == ineur-1) 
		{
			j = 0;
			i++;
		}
	}
		
	i = 0;	
	n = fs["in_wgain"];
	it = n.begin();
	it_end = n.end();
	for (; it != it_end; ++it)
	{
		in_wgain[i] = (float)*it;
		i++;
	}
	
	fs.release();
}
