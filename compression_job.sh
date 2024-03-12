#!/bin/sh

set -e


bold=$(tput bold)

Help()
{
   echo
   echo ${bold}"\033[31malgorithm_name\033[0m options:"
   echo
   echo ${bold}"\t    \e[5m\e[100mOffline\033[0m"
   echo
   echo "\t    Douglas-Peucker"
   echo "\t    Time-Ratio"
   echo "\t    Speed-Based"
   echo "\t    Heading-Based"
   echo "\t    Time-Speed-Heading"
   echo
   echo ${bold}"\t    \e[5m\e[100mOnline\033[0m"
   echo
   echo "\t    Dead-Reckoning"
   echo "\t    STTrace"
   echo
   echo ${bold}"\033[31mprovided_dataset\033[0m options:"
   echo
   echo "\t    Provide the absolute path of the dataset (csv)"
   echo
}



while getopts ":h" option; do
   case $option in
      h) # display Help
         Help
         exit;;
     \?) # incorrect option
         echo "Error: Invalid option"
         exit;;
   esac
done


CURDIR=`pwd`'/'

echo
echo -n "Please choose the mode of the compression:"
echo
echo
echo "1. Offline (of)"
echo "2. Online (on)"
echo
    read choice
    case $choice in
    of)
     echo ${bold}"\t    \033[31mAvailable choices\033[0m"
     echo
     echo "\t    Douglas-Peucker"
     echo "\t    Time-Ratio"
     echo "\t    Speed-Based"
     echo "\t    Heading-Based"
     echo "\t    Time-Speed-Heading"
    PATHTOCOMPRESSION='Offline/'
    ;;
    on)
     echo ${bold}"\t    \033[31mAvailable choices\033[0m"
     echo
     echo "\t    Dead_Reckoning"
     echo "\t    STTrace"
    PATHTOCOMPRESSION='Online/'
    ;;
    *)
    echo -n "Please provide a correct choice (of for Offline - on for Online)"
    echo
    exit;;
    esac


echo
echo -n "Please provide a) the name of the compression algorithm (choose from above list) and b) the absolute path of the dataset you want to employ."
echo
echo -n "run '\e[41msh compression_job.sh -h\033[0m' for more information"
echo
echo
read -p "Provide the compression algorithm name: " compre_algo
ALGORITHM=$compre_algo
read -p "Provide the absolute path of the dataset (csv): " dataset_tocomp
DATASET=$dataset_tocomp


PYTHONENDING='.py'


echo
echo ${bold}'Shell Command:'$CURDIR$PATHTOCOMPRESSION"\033[31m"$ALGORITHM"\033[0m"$PYTHONENDING ${bold}"\033[31m"$DATASET"\033[0m"
echo



if [ -f $CURDIR$PATHTOCOMPRESSION$ALGORITHM$PYTHONENDING -a -f $DATASET ]
then
  echo
  python3 $CURDIR$PATHTOCOMPRESSION$ALGORITHM$PYTHONENDING $DATASET
else
  echo "Algorithm or dataset does not exist"
  echo "Please provide one of the permitted compression algorithms (Maybe a typo?) or check if the provided dataset exists (check your path)"
  echo
  echo "run 'sh compression_job.sh -h' for more information"
fi