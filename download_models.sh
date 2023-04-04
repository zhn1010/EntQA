gdown https://drive.google.com/file/d/188_R29A3G3b2zuUs1oSRV2iSUmUomLpg/view --fuzzy -O ./models/data/kb # entities_kilt.json
gdown https://drive.google.com/file/d/1znMYd5HS80XpLpvpp_dFkQMbJiaFsQIn/view --fuzzy -O ./models # candidate_embeds.npy
gdown https://drive.google.com/file/d/1bHS5rxGbHJ5omQ-t8rjQogw7QJq-qYFO/view --fuzzy -O ./models # retriever.pt
gdown https://drive.google.com/file/d/1A4I1fJZKxmROIE1fd0mdXN6b1emP_xt4/view --fuzzy -O ./models # reader.pt
wget http://dl.fbaipublicfiles.com/BLINK/biencoder_wiki_large.json -P ./models # biencoder_wiki_large.json