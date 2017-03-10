FROM python:3.6-alpine
MAINTAINER James Endicott <james.endicott@colorado.edu>

ENV NUMPY_VERSION="1.11.2" \
    OPENBLAS_VERSION="0.2.18"
WORKDIR /app
ENTRYPOINT ["/bin/sh", "-c", "source /app/sh/entrypoint.sh"]

RUN export NPROC=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || 1) \
    && echo "http://alpine.gliderlabs.com/alpine/v3.4/main" > /etc/apk/repositories \
    && echo "http://alpine.gliderlabs.com/alpine/v3.4/community" >> /etc/apk/repositories \
    && echo "@edge http://alpine.gliderlabs.com/alpine/edge/community" >> /etc/apk/repositories \
    && apk --no-cache add openblas-dev@edge redis \
    && apk --no-cache add --virtual build-deps \
        g++ \
        linux-headers \
        musl-dev \
        openssl \
    && cd /tmp \
    && ln -s /usr/include/locale.h /usr/include/xlocale.h \
    && pip install cython \
    && wget http://downloads.sourceforge.net/project/numpy/NumPy/$NUMPY_VERSION/numpy-$NUMPY_VERSION.tar.gz \
    && tar -xzf numpy-$NUMPY_VERSION.tar.gz \
    && rm numpy-$NUMPY_VERSION.tar.gz \
    && cd numpy-$NUMPY_VERSION/ \
    && cp site.cfg.example site.cfg \
    && echo -en "\n[openblas]\nlibraries = openblas\nlibrary_dirs = /usr/lib\ninclude_dirs = /usr/include\n" >> site.cfg \
    && python -q setup.py build -j ${NPROC} --fcompiler=gfortran install \
    && cd /tmp \
    && rm -r numpy-$NUMPY_VERSION \
    && pip install \
        gensim \
        hiredis \
        nltk \
        redis \
        scipy \
        pyyaml \
    && python -m nltk.downloader -d /usr/share/nltk_data punkt wordnet stopwords \
    && apk --no-cache del --purge build-deps

COPY ./ /app/