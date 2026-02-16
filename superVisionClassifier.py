import model
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
from game import Environment
import os

def parser(str):
    charMap = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    moveList=[]
    for i in range(0, len(str), 2):
        moveList.append((charMap[str[i]], int(str[i+1]) - 1))
    return moveList

class supervisedDataloader (Dataset):
    
    def __init__(self, csv_file):
        if os.path.exists('./top_games_data_asTensor.pt'):
            pass
        else:
            csvData = pd.read_csv(csv_file)
            totalGameNum = len(csvData)
            maxPossibleMoves = totalGameNum * 60    #allocating ram for max possible moves
            allStates = torch.zeros((maxPossibleMoves, 3, 8, 8), dtype=torch.float32)
            allActions = torch.zeros((maxPossibleMoves,), dtype=torch.long)
            allWinners = torch.zeros((maxPossibleMoves,), dtype=torch.float32)
            env = Environment()
            currIdx = 0

            for i, row in csvData.iterrows():
                if i % 1000 == 0:
                    print(f"Processing game {i}/{totalGameNum}...")
                game = parser(row['game_moves'])
                truWinner = row['winner']

                for move in game:
                    legals = env.game.get_legal_moves()
                    if len(legals) == 0:
                        env.game.current_turn *= -1
                        legals = env.game.get_legal_moves()
                        if len(legals) == 0:
                            break
                    if move in legals:
                        
                        action = move[0] * 8 + move[1]
                        if env.game.current_turn == truWinner:
                            currWinner = 1
                        elif env.game.current_turn == -truWinner:
                            currWinner = -1
                        else:
                            currWinner = 0
                        
                        allStates[currIdx] = env.get_state()
                        allActions[currIdx] = action
                        allWinners[currIdx] = currWinner
                        currIdx += 1
                        
                        env.step(action)
                env.reset()

            finalAllStates = allStates[:currIdx]
            final_actions = allActions[:currIdx]
            final_winners = allWinners[:currIdx]
            torch.save({
                'states': finalAllStates,
                'actions': final_actions,
                'winners': final_winners
            }, './top_games_data_asTensor.pt')
            del allStates, allActions, allWinners
            print('data saved')

        
        self.data = torch.load('./top_games_data_asTensor.pt')

        '''
        will store data as boards state tensor, move made. For now only considering winning player's moves
        '''

    def __len__(self):
        return len(self.data['states'])

    def __getitem__(self, idx):        
        return self.data['states'][idx], self.data['actions'][idx], self.data['winners'][idx]

        
othelloDataLoader = supervisedDataloader('./othello_dataset.csv')
trainLoader = torch.utils.data.DataLoader(othelloDataLoader, batch_size=256, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = model.OthelloPlayer().to(device)

# criterionValue = nn.MSELoss()
# criterionAdvantage = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.0001)   

for epoch in range(3):
    resnet.train()
    for i, (states, actions, winners) in enumerate(trainLoader):
        states, actions, winners = states.to(device), actions.to(device), winners.to(device).float().view(-1, 1)
        winMask = (winners > 0).flatten()
        optimizer.zero_grad()

        if winMask.any():

            outputs = resnet(states[winMask], supervised = True)
            loss = criterion(outputs[winMask], actions[winMask])
            totalLoss = loss
            totalLoss.backward()
            optimizer.step()
        

        
        if i % 50 == 0:
            print(f"Epoch {epoch} | Batch {i} | Total Loss: {totalLoss.item():.4f}")

# Save the model after training
torch.save(resnet.state_dict(), 'othello_supervised.pth')
print("Training Complete. Model saved.")

