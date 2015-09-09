#include <opencv2/core/core.hpp>

#ifndef DATAMAN_H
#define DATAMAN_H

class Dataman
{
private:
	cv::FileStorage fs;
	int ineur;
	int oneur;
	int hneur;
public:
	std::vector< std::vector < float > > in_weight;
	std::vector< float > in_wgain;
	std::vector< std::vector < float > > out_weight;
	std::vector< float > out_wgain;
	std::vector< std::vector < float > > hid_weight;
	std::vector< float > hid_wgain;
	void write();
	void read();
	Dataman(int nneuron_input,int nneuron_output,int nneuron_hidden);
	~Dataman();
};

#endif 
