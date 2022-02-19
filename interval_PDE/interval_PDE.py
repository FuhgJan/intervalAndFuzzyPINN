import numpy as np
import torch
import torch.nn as nn
import copy
from torch.autograd import grad
import matplotlib.pyplot as plt



dev = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available, running on GPU")
    dev = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("CUDA not available, running on CPU")

torch.manual_seed(2021)


class MultiLayerNet(torch.nn.Module):
    def __init__(self, inp, out, activation, num_hidden_units=100, num_layers=1):
        super(MultiLayerNet, self).__init__()
        self.fc1 = torch.nn.Linear(inp, num_hidden_units, bias=True)
        self.fc2 = torch.nn.ModuleList()
        for i in range(num_layers):
            self.fc2.append(torch.nn.Linear(num_hidden_units, num_hidden_units, bias=True))
        self.fc3 = torch.nn.Linear(num_hidden_units, out, bias=True)
        self.activation = activation

        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-6)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500000, gamma=0.8)
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        for fc in self.fc2:
            x = fc(x)
            x = self.activation(x)
        x = self.fc3(x)
        return x


def kMinFun(x):
    return 100*(0.005*torch.sin(3*x[:,0:1]) * torch.cos(x[:,1:2])) #torch.cos(3*x) + 2.0

def kMaxFun(x):
    return 100*(0.01*torch.sin(3*x[:,0:1]) * torch.pow(torch.cos(x[:,1:2]),2)+0.03) #torch.cos(3.8*x) + 3.0

class MultiLayerNetE(torch.nn.Module):
    def __init__(self, inp, out, activation, num_hidden_units=100, num_layers=1):
        super(MultiLayerNetE, self).__init__()
        self.fc1 = torch.nn.Linear(inp, num_hidden_units, bias=True)
        self.fc2 = torch.nn.ModuleList()
        for i in range(num_layers):
            self.fc2.append(torch.nn.Linear(num_hidden_units, num_hidden_units, bias=True))
        self.fc3 = torch.nn.Linear(num_hidden_units, out, bias=True)
        self.activation = activation
        self.sig = torch.nn.Sigmoid()


        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-6)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=500000, gamma=0.8)

    def forward(self, xorig):
        x = self.fc1(xorig)
        x = self.activation(x)
        for fc in self.fc2:
            x = fc(x)
            x = self.activation(x)
        x = self.sig(self.fc3(x))

        EA_pred = (kMaxFun(xorig[:,0:2] ) - kMinFun(xorig[:,0:2] ))*x[:,0:2] + kMinFun(xorig[:,0:2] )


        return EA_pred


class IPINNnet(torch.nn.Module):
    def __init__(self, inp, out, activation, num_hidden_units=100, num_layers=1):
        super(IPINNnet, self).__init__()
        self.netu = MultiLayerNet(inp=inp, out=out, activation=activation, num_hidden_units=40, num_layers=3)
        self.netE = MultiLayerNetE(inp=inp, out=2, activation=activation, num_hidden_units=40, num_layers=3)
        self.loss_func = torch.nn.MSELoss()



    def train_model(self, x,x_0,u0,x_BC,uB, EPOCH):
        x = torch.from_numpy(x).float()
        x = x.to(dev)
        x.requires_grad_(True)

        u0 = torch.from_numpy(u0).float()
        u0 = u0.to(dev)
        u0.requires_grad_(True)

        uB = torch.from_numpy(uB).float()
        uB = uB.to(dev)
        uB.requires_grad_(True)


        state = copy.deepcopy(self.state_dict())
        best_loss = np.inf


        x_0 = torch.from_numpy(x_0).float()
        x_0 = x_0.to(dev)
        x_0.requires_grad_(True)

        x_BC = torch.from_numpy(x_BC).float()
        x_BC = x_BC.to(dev)
        x_BC.requires_grad_(True)



        for t in range(EPOCH):


            uout = self.get_u(x)
            k = self.netE(x)

            lossUMin = torch.sum(torch.abs(uout[:, 0]))
            lossUMax = -torch.sum( torch.abs(uout[:, 1]))


            duxut_min= grad(uout[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev),
                          create_graph=True, retain_graph=True)[0]
            duxux_min= grad(duxut_min[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev),
                          create_graph=True, retain_graph=True)[0]


            duxut_max= grad(uout[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev),
                          create_graph=True, retain_graph=True)[0]
            duxux_max= grad(duxut_max[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev),
                          create_graph=True, retain_graph=True)[0]

            Internal_min = duxut_min[:,1:2] - 0.01*uout[:,0:1] *duxux_min[:,0:1]  +  k[:,0:1]*  torch.pow(uout[:,0:1],3)-torch.pow(k[:,0:1],3)
            Internal_max = duxut_max[:,1:2] - 0.01* uout[:,1:2] *duxux_max[:,0:1]  +  k[:,1:2]*  torch.pow(uout[:,0:1],3) - torch.pow(k[:,1:2],3)


            #Internal_min = duxut_min[:,1:2] - 0.01*uout[:,0:1] *duxux_min[:,0:1]  +  netE2(x)[:,0:1]*  torch.pow(uout[:,0:1],3)-torch.pow(netE2(x)[:,0:1],3)       #duxut[:,1:2] -  (2* uout*duxut[:,0:1]   + torch.pow(uout[:,0:1],2) * duxux[:,0:1])     #-duxux[:,0:1] #-torch.pow(duxut[:,0:1],2) - uout* duxux[:,0:1]
            #Internal_max = duxut_max[:,1:2] - 0.01* uout[:,1:2] *duxux_max[:,0:1]  +  netE2(x)[:,0:1]*  torch.pow(uout[:,0:1],3) - torch.pow(netE2(x)[:,0:1],3)

            lossInternal_min = self.loss_func(Internal_min, torch.zeros(Internal_min.shape))
            lossInternal_max = self.loss_func(Internal_max, torch.zeros(Internal_max.shape))
            lossInternal = lossInternal_min + lossInternal_max

            # Boundary Loss
            uBC_pred = self.get_u(x_BC)
            LossBCu_min = self.loss_func( uBC_pred[:, 0], torch.zeros(uBC_pred[:, 0].shape))
            LossBCu_max = self.loss_func( uBC_pred[:, 1], torch.zeros(uBC_pred[:, 1].shape))

            boundary_loss =LossBCu_min + LossBCu_max


            # Initial Loss
            u0_pred = self.get_u(x_0)
            Loss0_min = self.loss_func( u0_pred[:, 0], u0[:, 0])
            Loss0_max = self.loss_func( u0_pred[:, 1], u0[:, 0])

            initial_loss =Loss0_min + Loss0_max


            # Total Loss
            nini = 1000000
            nint = 1000000

            loss = nint*(lossInternal) + nini*(initial_loss) + lossUMin + lossUMax



            self.netu.optimizer.zero_grad()  # clear gradients for next train

            self.netE.optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            self.netu.optimizer.step()  # apply gradients
            self.netE.optimizer.step()  # apply gradients

            self.netu.scheduler.step()
            self.netE.scheduler.step()

            if t%250==0:
                print('Iter: %d Loss: %.9e Internal: %.9e Boundary: %.9e  Initial: %.9e umin: %.9e umax: %.9e'
                      % (t + 1, loss.item(),nint*lossInternal.item(), boundary_loss.item(), nini*initial_loss.item(), lossUMin.item(), lossUMax.item()))
            if t % 5000 == 0:
                nt = 11
                lin_test = np.linspace(0, 1.0, nt).reshape(nt, 1)

                for ii in range(nt):
                    self.evaluate_model(lin_x, lin_test[ii],t)

                torch.save(self.netE.state_dict(), '../output/modelsDuringTrain/NetEAbsSaveWithOutMin.pt')
                torch.save(self.netu.state_dict(), '../output/modelsDuringTrain/NetuAbsSaveWithoutMin.pt')


        self.load_state_dict(state)




    def evaluate_model(self, x,t, Iteration):
        dom = np.array(np.meshgrid(x, t)).T.reshape(-1,2)
        dom= torch.from_numpy(dom).float()
        dom = dom.to(dev)
        dom.requires_grad_(True)


        u = self.get_u(dom)
        k = self.netE(dom)
        kmin = kMinFun(dom)
        kmax = kMaxFun(dom)

        u = u.cpu().detach().numpy()
        dom = dom.cpu().detach().numpy()
        k = k.cpu().detach().numpy()
        kmin = kmin.cpu().detach().numpy()
        kmax = kmax.cpu().detach().numpy()


        ts = str(t[0])
        ts = ts.translate({ord('.'): None})
        ts = ts[0:2]

        f, ax = plt.subplots()

        plt.title('Time = '+ str(t[0])[0:3]+ 's')
        ax.plot(dom[:,0], u[:,0],label=r'$u_{min}$', color='black')
        ax.plot(dom[:,0], u[:,1],label=r'$u_{max}$', color='red')
        plt.grid(b=True, which='major', color='black', linestyle=':', alpha=0.5)
        #plt.ylim([-1,3])
        #ax.plot(x, y[:,1],label= 'uMax')
        ax.legend()
        St = '../output/images/imagesDuringTraining/'+str(Iteration) + '_WithFoundMinstochDiffusion_'+ts+'.png'
        plt.savefig(St)
        plt.close()


        f, ax = plt.subplots()
        plt.title('Time = '+ str(t[0])[0:3] + 's')
        ax.plot(dom[:,0], k[:,0],label=r'$k_{min}$', color='black')
        ax.plot(dom[:,0], k[:,1],label=r'$k_{max}$', color='red')
        ax.plot(dom[:,0], kmin,label=r'$k_{lb}$', linestyle='--', color='black')
        ax.plot(dom[:,0], kmax,label=r'$k_{ub}$',linestyle='--', color='red')
        plt.grid(b=True, which='major', color='black', linestyle=':', alpha=0.5)
        #plt.ylim([-1,3])
        #ax.plot(x, y[:,1],label= 'uMax')
        ax.legend()
        St = '../output/images/imagesDuringTraining/'+str(Iteration) + '_WithFoundMinstochDiffusionEA_'+ts+'.png'
        plt.savefig(St)
        plt.close()



    def get_u(self, x):

        u = self.netu(x)
        return  (x[:,0:1]+1) * (x[:,0:1] - 1)*u


    # --------------------------------------------------------------------------------
    # purpose: loss square sum for the boundary part
    # --------------------------------------------------------------------------------
    @staticmethod
    def loss_squared_sum(tinput, target):
        row, column = tinput.shape
        loss = 0
        for j in range(column):
            loss += torch.sum((tinput[:, j] - target[:, j]) ** 2) / tinput[:, j].data.nelement()
        return loss


if __name__ == '__main__':
    # ----------------------------------------------------------------------
    #                   STEP 1: SETUP DOMAIN
    # ----------------------------------------------------------------------
    nx = 125
    lin_x = np.linspace(-1, 1, nx).reshape(nx,1)

    nt = 50
    lin_t = np.linspace(0, 1, nt).reshape(nt,1)

    dom = np.array(np.meshgrid(lin_x, lin_t)).T.reshape(-1,2)

    # Initial condition
    x0 = np.zeros((nx,2))
    x0[:,0:1] = lin_x
    u0 = 1 - np.power(lin_x,2)
    # Boundary condition
    lin_x_XB = np.linspace(-1, 1, 2).reshape(2,1)
    xB = np.array(np.meshgrid(lin_x_XB, lin_t)).T.reshape(-1,2)

    uB = np.zeros((2,1))
    uB[0] = 0.0
    uB[1] = 0.0
    # ----------------------------------------------------------------------
    #                   STEP 2: SETUP MODEL
    # ----------------------------------------------------------------------
    iPINN = IPINNnet(inp=2, out=2, activation=nn.Tanh(), num_hidden_units=30, num_layers=3)

    usePreTrainedModels = True

    if not usePreTrainedModels:
        iPINN.train_model(dom,x0,u0,xB, uB,500000)

    if usePreTrainedModels:
        netu = iPINN.netu
        netE = iPINN.netE

        nt = 11
        lin_test = np.linspace(0, 1.0, nt).reshape(nt,1)


        for ii in range(nt):
            dom = np.array(np.meshgrid(lin_x,lin_test[ii])).T.reshape(-1, 2)
            dom = torch.from_numpy(dom).float()
            dom = dom.to(dev)
            dom.requires_grad_(True)


            netE.load_state_dict(torch.load('../output/preTrainedModels/NetEAbsSave.pt', map_location=dev))
            netu.load_state_dict(torch.load('../output/preTrainedModels/NetuAbsSave.pt', map_location=dev))
            uPINN = iPINN.get_u(dom)
            kPINN = iPINN.netE(dom)
            uPINN = uPINN.cpu().detach().numpy()

            kmin = kMinFun(dom)
            kmax = kMaxFun(dom)
            kPINN = kPINN.cpu().detach().numpy()
            kmin = kmin.cpu().detach().numpy()
            kmax = kmax.cpu().detach().numpy()
            domnp = dom.cpu().detach().numpy()
            ts = str(lin_test[ii][0])
            ts = ts.translate({ord('.'): None})
            ts = ts[0:2]

            f, ax = plt.subplots()


            ax.plot(domnp[:, 0], uPINN[:, 0], label=r'$\hat{u}^{min}$', color='black',Linewidth=3.0)
            ax.plot(domnp[:, 0], uPINN[:, 1], label=r'$\hat{u}^{max}$', color='red',Linewidth=3.0)
            plt.grid(b=True, which='major', color='black', linestyle=':', alpha=0.5)
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.xlabel(r'$x$',fontsize=14)
            plt.ylabel(r'$u$',fontsize=14)
            ax.legend()
            St = '../output/images/' + str('Final') + '_stochDiffusion_' + ts + '.pdf'
            plt.savefig(St)
            plt.close()

            f, ax = plt.subplots()
            ax.plot(domnp[:,0], kPINN[:,0],label=r'$\hat{k}^{min}$', color='black')
            ax.plot(domnp[:,0], kmax,label=r'$\hat{k}^{max}$', color='red')
            ax.plot(domnp[:,0], kmin,label=r'$k^{L}$', linestyle='--', color='black',LineWidth=3.0)
            ax.plot(domnp[:,0], kmax,label=r'$k^{U}$',linestyle='--', color='red',LineWidth=3.0)
            plt.grid(b=True, which='major', color='black', linestyle=':', alpha=0.5)
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.xlabel(r'$x$',fontsize=14)
            plt.ylabel(r'$k$',fontsize=14)
            ax.legend(loc='right',ncol=2)
            St = '../output/images/'+str('Final') + '_stochDiffusionEA_'+ts+'.pdf'
            plt.savefig(St)
            plt.close()



            netE.load_state_dict(torch.load('../output/preTrainedModels/NetEAbsSaveWithOutMin.pt', map_location=dev))
            netu.load_state_dict(torch.load('../output/preTrainedModels/NetuAbsSaveWithoutMin.pt', map_location=dev))
            uMin = iPINN.get_u(dom)
            kMIN = iPINN.netE(dom)
            uMin = uMin.cpu().detach().numpy()

            dom = dom.cpu().detach().numpy()

            ts = str(lin_test[ii][0])
            ts = ts.translate({ord('.'): None})
            ts = ts[0:2]

            f, ax = plt.subplots()
            ax.plot(dom[:, 0], uMin[:, 0], label=r'$\hat{u}^{min}$ with $\hat{k}^{min}$', color='black',Linewidth=2.0)
            ax.plot(dom[:, 0], uPINN[:, 0], label=r'$\hat{u}^{min}$ with $k^{L}$', color='red',Linewidth=2.0)
            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.grid(b=True, which='major', color='black', linestyle=':', alpha=0.5)
            plt.xlabel(r'$x$',fontsize=14)
            plt.ylabel(r'$u$',fontsize=14)
            ax.legend()
            St = '../output/images/' + str('FinalCOMPWithkMIN') + '_MinstochDiffusion_' + ts + '.pdf'
            plt.savefig(St)
            plt.close()



