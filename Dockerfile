FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    tmux \
    nano \
    htop \
    wget \
    curl \
    git \
    libsm6 \        
    libxrender1 \  
    libfontconfig1 \ 
    ffmpeg \
    libxext6 \
    openssh-server \
    cmake \
    libncurses5-dev \
    libncursesw5-dev \
    build-essential

RUN echo 'PermitRootLogin yes\nSubsystem sftp internal-sftp\nX11Forwarding yes\nX11UseLocalhost no\nAllowTcpForwarding yes' > /etc/ssh/sshd_config
EXPOSE 22
RUN groupadd sshgroup
RUN mkdir /var/run/sshd
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh
COPY cm-docker.pub /root/.ssh
RUN cat /root/.ssh/cm-docker.pub >> /root/.ssh/authorized_keys
RUN echo 'PATH=$PATH:/opt/conda/bin' >> ~/.bashrc # somehow conda is missing from PATH if login via ssh

RUN echo 'root:f5n6Nn5F@aMA' | chpasswd 
# REPLACE f5n6Nn5F@aMA WITH YOUR PASSWORD

# Force bash color prompt
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/g' ~/.bashrc

RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh \n" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

COPY requirements.txt /workspace
RUN ["conda", "run", "-n", "base", "pip", "install", "-r", "/workspace/requirements.txt"]

CMD ["/bin/bash"]
