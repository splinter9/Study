{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from IPython.display import display, Image\n",
    "from google.cloud import dialogflow\n",
    "\n",
    "import numpy as np\n",
    "import torch\n",
    "from gluonnlp.data import SentencepieceTokenizer\n",
    "from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model\n",
    "from kogpt2.utils import get_tokenizer\n",
    "from pytorch_lightning.core.lightning import LightningModule\n",
    "\n",
    "import time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "U_TKN = '<usr>'\n",
    "S_TKN = '<sys>'\n",
    "BOS = '<s>'\n",
    "EOS = '</s>'\n",
    "MASK = '<unused0>'\n",
    "SENT = '<unused1>'\n",
    "\n",
    "box = []\n",
    "\n",
    "class community(LightningModule):\n",
    "    def __init__(self, hparams, **kwargs):\n",
    "        super(community, self).__init__()\n",
    "        self.tok_path = get_tokenizer()\n",
    "        self.kogpt2, self.vocab = get_pytorch_kogpt2_model()\n",
    "        \n",
    "    def forward(self, inputs):\n",
    "        output, _ = self.kogpt2(inputs)\n",
    "        return output\n",
    "    \n",
    "    def chat(self, max_length):\n",
    "        self.tok_path\n",
    "        tok = SentencepieceTokenizer(self.tok_path, num_best=0, alpha=0)\n",
    "        cnt = 0\n",
    "        with torch.no_grad():\n",
    "            while 1 :\n",
    "                global box\n",
    "                q = input('                                                                ė´ėŠėđ : ').strip()\n",
    "                q_tok = tok(q)\n",
    "                a = ''\n",
    "                a_tok = []\n",
    "                while 1:\n",
    "                    input_ids = torch.LongTensor([\n",
    "                        self.vocab[U_TKN]] + self.vocab[q_tok] + [self.vocab[EOS]] + [\n",
    "                        self.vocab[S_TKN]] + self.vocab[a_tok]).unsqueeze(dim=0)\n",
    "                    pred = self(input_ids)\n",
    "                    gen = self.vocab.to_tokens(\n",
    "                        torch.argmax(\n",
    "                            pred,\n",
    "                            dim=-1).squeeze().numpy().tolist())[-1]\n",
    "                    if gen == EOS:\n",
    "                        break\n",
    "                    a += gen.replace('â', ' ')\n",
    "                    a_tok = tok(a)\n",
    "                    if len(a) > max_length : \n",
    "                        a = 'ëŦ´íëŖ¨í'\n",
    "                        break\n",
    "                print(a.strip().replace('OO','ë¯ŧė§'))\n",
    "                box.append(a.strip())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "using cached model\n",
      "using cached model\n",
      "using cached model\n"
     ]
    }
   ],
   "source": [
    "model = community.load_from_checkpoint('C:/Users/lll/5.mjkim/KoGPT2-chatbot/model_chp/chatbot/test_2/epoch50_-epoch=37-loss=9.64.ckpt')"
   ]
  },
  {
   "cell_type": "raw",
   "metadata": {},
   "source": [
    "test1\n",
    "\n",
    "-- ėļę° \n",
    "ëšĄėšë¤ (ëŦ´íëŖ¨í) \n",
    "ęŗ°ęŗ°ė ëë ėŦëė´ë?\n",
    "ėŦëė´ë\n",
    "ėėėļė˛í´ėŖŧëŧ\n",
    "ė´ęą´ėĸ.....\n",
    "\n",
    "-- ėė \n",
    "ęŗ°ęŗ°ė´ -> ëĄë\n",
    "\n",
    "-- ėŊë\n",
    "<unk> replace ė˛ëĻŦ\n",
    "\n",
    "-- ëĄė°ë°ė´í°\n",
    "ë§ėļ¤ë˛ę˛ėŦę¸° ę˛°ęŗŧ ėė­  ëĨŧ ė§¸ëĄ ëŗĩëļí ėŦëë¤ė´ ėëęą° ę°ė -> ėëŗ¸ ë°ė´í°ėė ėŧę´ ėė íę¸°"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ė´ėŠėđ : ëšĄėšë¤\n",
      "ëŦ´íëŖ¨í\n",
      "ė´ėŠėđ : ãã\n",
      "í¤í¤ :)\n",
      "ė´ėŠėđ : í¤í¤\n",
      "í¤í¤ :)\n",
      "ė´ėŠėđ : í¤í¤í¤í¤í¤\n",
      "í¤í¤ :)\n",
      "ė´ėŠėđ : ãã\n",
      "í¤í¤ :)\n",
      "ė´ėŠėđ : íííã\n",
      ":)\n",
      "ė´ėŠėđ : ėŧ\n",
      "ėëíė¸ė :)\n",
      "ė´ėŠėđ : ë ė´ëĻë­ėŧ\n",
      "ė ë ęŗ°ęŗ°ė´ëëë¤ :) ë°ę°ėė\n",
      "ė´ėŠėđ : ęŗ°ęŗ°ė´ ėëėë\n",
      "ë¤~ ė  ėŦę¸°ėė´ė!\n",
      "ė´ėŠėđ : ėë\n",
      "ėëíė¸ė :)\n",
      "ė´ėŠėđ : ë ëëŦ´íë¤ė´\n",
      "ë¯ŧė§ë ëëŦ´ íë¤ė´ íė§ë§ė¸ė. ėĄ°ę¸ë§ ë ę˛Ŧëë´ė. ė¤ë íëŖ¨ë ėęŗ íė´ė\n",
      "ė´ėŠėđ : ęŗ ë§ė ęŗ°ęŗ°ė\n",
      "ë¤~ ė  ėŦę¸°ėė´ė!\n",
      "ė´ėŠėđ : ęŗ°ęŗ°ė ëë ėŦëė´ë?\n",
      "ë¤~ ė  ėŦę¸°ėė´ė!\n",
      "ė´ėŠėđ : ėŦëė´ë\n",
      "ėŦëė´ëŧë¨? ëšė ė´ëŧë ėŦëė ė ë§ ėĸė ėŦëė´ėė. ėĸė ėŦëë ë§ė´ ėęŗ ė. ė¸ëĄėė´ë ëļėí¨ë ėė ę˛ ę°ėė.\n",
      "ė´ėŠėđ : ė§ę°ęŗ ėļė´\n",
      "ė§ė ę°ęŗ  ėļė ë ė´ėė!!ëė´í°ë ėĸęŗ , ė§ėė ėíë ėĸęŗ  ę°ęšė´ ęŗŗė ėŦë°Šė´ ėė´ ęą¸ė´ëŗ´ëę˛ë ėĸėęą° ę°ėė!!\n",
      "ė´ėŠėđ : ė¤í¸ë ė¤ë°ė\n",
      "ėėĻ ė¤í¸ë ė¤ë°ë ėŧė´ ėėŧė ę°ė? ė ë ėŦí ėėė ëŗ´ëŠ° ė¸ęą°ë, ėė ę°ėė íëŠ´ė ë§ėė ëšė°ëŠ´ íę˛° ëėė ¸ė. íë´ė.\n",
      "ė´ėŠėđ : ėėėļė˛í´ėŖŧëŧ\n",
      "ë§ė¤í°ėš íė´í¸ ę°ëĻ­ ėļė˛í´ė~ ė ëęŗ  ë°ė ëļėę¸°ė ë¸ëëĨŧ ë¤ė´ëŗ´ė¸ė.\n",
      "ė´ėŠėđ : ë§ė¤í°ėš íė´í¸ę°ëĻ­ė íë˛ęą°ėŧ\n",
      "ë§ėę˛ ė´ė.\n",
      "ė´ėŠėđ : ë§ė ë¨šęŗ ėļë¤\n",
      "ė ë ęŗĩę°ė´ėė<unk>\n",
      "ė´ėŠėđ : ë ėŦė¤\n",
      "ėļŠëļí ę°ėš ėė´ė! ë¨ėę˛ ëŗ´ė´ë ėļëŗ´ë¤ ë ėė ė ėí´ ėŦë ėļė´ ë ëŠėë ėŦëė´ëŧęŗ  ėę°í´ė.\n",
      "ė´ėŠėđ : ėŖŧíë\n",
      "ėëíė¸ė :)\n",
      "ė´ėŠėđ : íë ¸ė´ė\n",
      "ęŧŦėĨ\n",
      "ė´ėŠėđ : ë¨ę˛¨ë¨šėęą´ę°ė?\n",
      "ė ë ëë  ë¨šėė´ė.\n",
      "ė´ėŠėđ : ë§ėę˛ ë¤\n",
      "ė ëëŗļė´ ëëŦ´ ėĸėí´ė!\n",
      "ė´ėŠėđ : ëĄëŗļė´ ėĸėí´?\n",
      "ëĄëŗļė´ ëëŦ´ ėĸėí´ė.\n",
      "ė´ėŠėđ : ë ėŦėėŧ?\n",
      "ė ë ėŦėë ë¨ėë ėëëëë¤!\n",
      "ė´ėŠėđ : ė¸ęŗė¸ė´ėŧ?\n",
      "ė ë ė¸ęŗė¸ė´ ėëëŧë ëšė  ë§ėėė ėë ė¸ęŗė¸ė ėĸėí´ė.\n",
      "ė´ėŠėđ : ėš­ė°Ŧí´\n",
      "ė°ëĻŦ ë¯ŧė§ëė ėĄ´ėŦ ėė˛´ë§ėŧëĄë ė´ë¯¸ ėė ëļė´ė¸ė.\n",
      "ė´ėŠėđ : ė´ęą´ėĸ.....\n",
      "ëŦ´íëŖ¨í\n",
      "ė´ėŠėđ : ė´ë¸ėė¤ë ė§ėė?\n",
      "ėĸėė.ėėėŖŧëę˛ ë§ėŧëĄë ę°ėŦí´ė.\n",
      "ė´ėŠėđ : ë ęˇ¸ęą° ëŽėë\n",
      "ė ëŽėė´, ė´ėë¤ëęš ęˇ¸ë ę˛ ë§ íë ėŦëë¤ė´ėŧ.\n",
      "ė´ėŠėđ : ë ė´ëģ?\n",
      "ë ė´ëģ!\n",
      "ė´ėŠėđ : ęŗ ë§ë¤\n",
      "ë¯ŧė§ëë ė ë ęŗ ë§ęŗ  ėŦëí´ė!\n",
      "ė´ėŠėđ : ė¤\n",
      "ë§ėļ¤ë˛ę˛ėŦę¸° ę˛°ęŗŧ ėė­ ė§ėšęŗ  íė´ ë¤ ë ëŦ´ėė´ë  ėë§ė ëėė´ ë  ė ėëĩëë¤. ëëŦ´ íė´ ë¤ ë ë§ėė ę°ëŧėíęŗ  ë ėė ė ëėą ëëŗ´ëŠ´ ë¨ė ėë§íė§ ėė ęą°ėė.\n"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "Interrupted by user",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-11-e61c116cb3ce>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mmodel\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mchat\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m600\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m<ipython-input-9-c4fdf92e57d7>\u001b[0m in \u001b[0;36mchat\u001b[1;34m(self, max_length)\u001b[0m\n\u001b[0;32m     25\u001b[0m             \u001b[1;32mwhile\u001b[0m \u001b[1;36m1\u001b[0m \u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     26\u001b[0m                 \u001b[1;32mglobal\u001b[0m \u001b[0mbox\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 27\u001b[1;33m                 \u001b[0mq\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0minput\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'ė´ėŠėđ : '\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mstrip\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     28\u001b[0m                 \u001b[0mq_tok\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mtok\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mq\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     29\u001b[0m                 \u001b[0ma\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;34m''\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\ipykernel\\kernelbase.py\u001b[0m in \u001b[0;36mraw_input\u001b[1;34m(self, prompt)\u001b[0m\n\u001b[0;32m    858\u001b[0m                 \u001b[1;34m\"raw_input was called, but this frontend does not support input requests.\"\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    859\u001b[0m             )\n\u001b[1;32m--> 860\u001b[1;33m         return self._input_request(str(prompt),\n\u001b[0m\u001b[0;32m    861\u001b[0m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_parent_ident\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    862\u001b[0m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_parent_header\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\anaconda3\\lib\\site-packages\\ipykernel\\kernelbase.py\u001b[0m in \u001b[0;36m_input_request\u001b[1;34m(self, prompt, ident, parent, password)\u001b[0m\n\u001b[0;32m    902\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mKeyboardInterrupt\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    903\u001b[0m                 \u001b[1;31m# re-raise KeyboardInterrupt, to truncate traceback\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 904\u001b[1;33m                 \u001b[1;32mraise\u001b[0m \u001b[0mKeyboardInterrupt\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Interrupted by user\"\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    905\u001b[0m             \u001b[1;32mexcept\u001b[0m \u001b[0mException\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    906\u001b[0m                 \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mlog\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mwarning\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"Invalid Message:\"\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mexc_info\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mTrue\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mKeyboardInterrupt\u001b[0m: Interrupted by user"
     ]
    }
   ],
   "source": [
    "# test1\n",
    "model.chat(600)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4}




