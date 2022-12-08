# EBC 7100 Project
Project - Good Food is Good Mood
=======
Authors:
1. Jiaxi Chen (jchen421@uottawa.ca)
2. Salauddin Ali Ahmed (salia023@uottawa.ca)
3. Vrushti Buch (vbuch104@uottawa.ca)
4. Saif Khalid Ahmadzai(sahma013@uottawa.ca)

The attached source code only contains one python script.
Pre-requisites:
1. Python 3
2. Any python IDE (we used Spider and Jupyter notebooks using Anaconda)
3. Docker Edge (For running web application)
4. flask_restful dependency (For API)

The code structure is as follows:

1. Importing libraries 
2. User defined functions
3. Data Preparation and Pre-processing
4. Vectorization
5. Cross Validation
6. Error Analysis

Steps to execute the script:
1. Install Python 3 (https://www.python.org/download/releases/3.0/) or use homebrew if you are a mac user
2. Ensure Python 3 installed properly using the following command 
3. install - python
4. To test if Python is successfully installed, type python on terminal/console. 
   "You should see the output as below:
    Python 3.7.2 (default, Dec 29 2018, 00:00:04) 
    [Clang 4.0.1 (tags/RELEASE_401/final)] :: Anaconda, Inc. on darwin
    Type "help", "copyright", "credits" or "license" for more information.
    >>>" 
6. Download the "assignment1".py file to your local machine
7. Open command prompt (Windows) or terminal (macintosh)
8. Enter the command - "python <path_of_the_downloaded py file>"
9. For the best results and proper vizualization of charts, we recommend using Spider or Jupyter notebook.
   Download the program and open it with Jupyter notebook and run the cell.  

Deliverables:
1. Project Report
2. Source Code along with the ReadMe File.
3. Source Code for Angular Application and Python API
4. Dockerfile (as a part of Angular Application root folder)
5. Results in HTML format
6. Presentation

Steps to use the web application:
1. Install DockerEdge from 
2. Make sure the Docker is running in your machine
3. Go to terminal and type Docker to make sure Docker is properly installed in your machine
4. Once the docker is properly installed, navigate to the cuisine-recomm folder saved in your drive
5. Build the docker image using the following command - 
	docker -t cuisine-recomm:v1 .
6. The Docker build will take a while. Once it is executed, run the docker image
	docker run -p 80:80 cuisine-recomm:v1
7. Once the Docker image is run, please follow the steps below to bring up the API server
8. Open "" file in any python ide and run the file
9. You will observe that the server is up at http://127.0.0.1:5002
10. Once the server is up, you can start using the web application
11. The results take a while to show up, you can see the execution log in the Python API IDE

For further questions, feel free to reach us at the e-mail address provided above.
