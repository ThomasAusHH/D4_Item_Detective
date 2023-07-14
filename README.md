## Install

* Clone this repository
* Go to the root of cloned repository
* Install dependencies by running `pip3 install -r requirements.txt`

## Run

Execute:

```
python3 Item_Checker.py
```


## FAQ
* Set Framerate of screenshots `framerate = 2`
* Set Monitor to capture `monitor_index = 1` 1 = Main 2 = Second
* After start you should see a new window that shows the stream of screenshots. It will highlight the found item stats ingame.
* It runs OCR for found item stats

## Example Output
![2023-07-15 00_58_43-Object Detection](https://github.com/ThomasAusHH/D4_Item_Detective/assets/75835669/02409a0a-b49d-4bf5-aa7b-b79254095b7e)

### Console output
* 0: 384x640 4 Item-Affixs, 2 Item-Affixess, 2 Item-Aspects, 3 Item-Powers, 2 Item-Tooltips, 433.0ms
* Speed: 8.9ms preprocess, 433.0ms inference, 6.9ms postprocess per image at shape (1, 3, 384, 640)
* Class: Item-Aspect, Probability: 0.95, OCR: ® You gain 0.50%|x] (0.25 - 0.50}%  increased Armor for 4 seconds when  you deal any form of damage, stacking  upto 50.00% x] [25.00 - 50.00%.  oe 
* Class: Item-Affixes, Probability: 0.95, OCR: +7.0% Healing Received [4.5 - 8.0) ¢  +7.0%)  20.5% Shadow Resistance [17.5 -  28.0) {+ 20.5%)  +25 Dexterity +[18 - 25] (+25)  +25 Intelligence +(18-25]( ) 
* Class: Item-Aspect, Probability: 0.92, OCR: ® Imprinted: Basic Skills grant 20%  Damage Reduction for 3.0 [2.0 - 6.0]  7 seconds.
* Class: Item-Power, Probability: 0.86, OCR: 564 Item Power
* Class: Item-Affixes, Probability: 0.49, OCR: © +31 Intelligence +[31 - 44]   +9.8% Barrier Generation [6.5 - 13.0}%  * 8.1% Cooklown Reduction [4.5 - 9.6]%  o +7.1% Basic Skill Attack Speed [4.5 -  9.8)%     
* Class: Item-Power, Probability: 0.44, OCR: 649+ 15 Item Power
