FROM python:3.8-slim-buster
LABEL maintainer "Dheeraj Madda <dheerajmadda@gmail.com>"
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# install opencv dependencies
RUN apt-get update && \
	apt-get install -y python3-opencv

# install necessary packages
RUN pip install --upgrade pip \
	&& pip install --no-cache-dir -r requirements.txt
	
RUN apt-get clean

# configure streamlit
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
	[general]\n\
	email = \"\"\n\
	" > /root/.streamlit/credentials.toml'
	
RUN bash -c 'echo -e "\
	[server]\n\
	enableCORS = false\n\
	" > /root/.streamlit/config.toml'
	

EXPOSE 8080
ENTRYPOINT ["streamlit", "run", "--server.port", "8080"] 
CMD ["app.py"] 
