from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.models import User
from django.contrib import messages
from django.core.mail import EmailMessage, send_mail
from predictly import settings
from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from django.contrib.auth.tokens import default_token_generator
from django.utils.encoding import force_bytes, force_str
from django.contrib.auth import authenticate, login, logout
from django.views.generic import View
from django.contrib.auth.decorators import login_required

from . tokens import generate_token
from rest_framework.views import APIView
from .forms import StockForm
from .models import Stock
from .forms import PasswordResetForm, FeedbackForm

from .constants import (
    FAQ_GLOSSARY, 
    MARKET_GLOSSARY
)

from .plotting import (
    generate_closing_prices_plot,
    generate_crossover_plot, 
    generate_prediction_vs_actual_plot,
    generate_weekly_forecast_plot,
    determine_cross_signal
)
from .services import (
    fetch_stock_data,
    get_company_info,
    calculate_sma_dataframe,
    prepare_lstm_data,
    get_or_train_lstm_model,
    make_predictions,
    generate_future_predictions
)

import pandas as pd
import datetime
import os
import requests
import json

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

def home(request):
    return render(request, "authentication/signup.html")

def password_reset_done(request):
    return render(request, "authentication/password_reset_done.html")

def password_reset_complete(request):
    return render(request, "authentication/password_reset_complete.html")

def faq(request):
    return render(request, "pages/faq.html", {"glossary": FAQ_GLOSSARY})

def market_glossary(request):
    fname = request.user.first_name if request.user.is_authenticated else ""
    context = {'fname': fname,'glossary': MARKET_GLOSSARY}
    return render(request, "pages/market_glossary.html", context)

def signup(request):
    if request.method == "POST":
        username = request.POST['username']
        fname = request.POST['fname']
        lname = request.POST['lname']
        email = request.POST['email']
        pass1 = request.POST['pass1']
        pass2 = request.POST['pass2']
        
        if User.objects.filter(username=username):
            messages.error(request, "Username already exist! Please try some other username.")
            return redirect('home')
        
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email Already Registered!!")
            return redirect('home')

        if len(username)>20:
            messages.error(request, "Username must be under 20 charcters!!")
            return redirect('home')
        
        if pass1 != pass2:
            messages.error(request, "Passwords didn't matched!!")
            return redirect('home')

        if not username.isalnum():
            messages.error(request, "Username must be Alpha-Numeric!!")
            return redirect('home')

        myuser = User.objects.create_user(username, email, pass1)
        myuser.first_name = fname
        myuser.last_name = lname
        myuser.is_active = False  
        myuser.save()
        messages.success(request, "Your Account has been created succesfully! Please check your email to confirm your email address in order to activate your account.")
        
        # Welcome Email Start
        subject = "🌟 Welcome to Predictly!"
        message = f"Hey {myuser.first_name}, welcome to Predictly! 🚀\n\nGet ready to ride the wave of stock trends with us. Your financial journey starts here!\n\nHowever, first things first, we've sent you a confirmation email, please check your inbox to confirm your email address.\n\nWe can't wait to see you delve into the stock world.\n\n- The Predictly Team 📈"

        from_email = settings.EMAIL_HOST_USER
        to_list = [myuser.email]

        html_message = f"""
            <html>
            <head>
                <style>
                    body {{
                        font-family: 'Arial', sans-serif;
                        margin: 20px;
                        padding: 20px;
                        background-color: #f4f4f4;
                        color: #333;
                    }}
                    .container {{
                        max-width: 600px;
                        margin: 0 auto;
                        background-color: #fff;
                        padding: 40px;
                        border-radius: 8px;
                        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    }}
                    h1 {{
                        color: #FF4B4B;
                    }}
                    p {{
                        margin-bottom: 20px;
                    }}
                    strong {{
                        font-weight: bold;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Welcome to Predictly!</h1>
                    <p>Hey <strong>{myuser.first_name}</strong>, welcome to Predictly! 🚀</p>
                    <p>Get ready to ride the wave of stock trends with us. Your financial journey starts here!</p>
                    <p>However, first things first, we've sent you a confirmation email, please check your inbox to confirm your email address.</p>
                    <p>We can't wait to see you delve into the stock world.</p>
                    <p>- The Predictly Team 📈</p>
                </div>
            </body>
            </html>
        """

        send_mail(
            subject,
            '',  
            from_email,
            to_list,
            html_message=html_message,  
            fail_silently=True,
        )
        
        # Welcome Email End

        # User Email Confirmation Start
        
        current_site = get_current_site(request)
        email_subject = "🚀 Confirm your Email @ Predictly"

        message2 = render_to_string('email_templates/user_email_confirmation.html', {
            'name': myuser.first_name,
            'domain': current_site.domain,
            'uid': urlsafe_base64_encode(force_bytes(myuser.pk)),
            'token': generate_token.make_token(myuser),
        })

        email = EmailMessage(
            email_subject,
            '',
            settings.EMAIL_HOST_USER,
            [myuser.email],
        )
        email.content_subtype = 'html'  
        email.body = message2  
        email.fail_silently = True
        email.send()
        
        # User Email Confirmation End
        
        return redirect('signin') 
        
    return render(request, "authentication/signup.html")

def signin(request):
    if request.method == 'POST':
        username = request.POST['username']
        pass1 = request.POST['pass1']
        
        user = authenticate(username=username, password=pass1)
        
        if user is not None:
            login(request, user)
            fname = user.first_name
            return redirect('last-trade-prices', ticker_symbol='AAPL')  

        else:
            messages.error(request, "Bad Credentials!!")
            return redirect('signin')

    return render(request, "authentication/signin.html")

def signout(request):
    logout(request)
    messages.success(request, "Logged Out Successfully!!")
    return redirect('home')

def activate(request,uidb64,token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        myuser = User.objects.get(pk=uid)
    except (TypeError,ValueError,OverflowError,User.DoesNotExist):
        myuser = None

    if myuser is not None and generate_token.check_token(myuser,token):
        myuser.is_active = True
        # user.profile.signup_confirmation = True
        myuser.save()
        login(request,myuser)
        messages.success(request, "Your Account has been activated!!")
        return redirect('signin')
    else:
        return render(request,'activation_failed.html')
    
def forgot_password(request):
    if request.method == 'POST':
        form = PasswordResetForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data['email']
            associated_users = User.objects.filter(email=data)
            if associated_users.exists():
                for user in associated_users:
                    token = default_token_generator.make_token(user)
                    uid = urlsafe_base64_encode(force_bytes(user.pk))
                    subject = '🔒 Password Reset Requested'
                    email_template_name = 'email_templates/reset_password_email.html'
                    c = {
                        'email': user.email,
                        'domain': request.META['HTTP_HOST'],
                        'site_name': 'your site',
                        'uid': uid,
                        'user': user,
                        'token': token,
                        'protocol': 'http',
                    }
                    email = render_to_string(email_template_name, c)
                    send_mail(subject, '', 'predictlyapp@gmail.com', [user.email], fail_silently=False, html_message=email)
                return redirect("password_reset_done")
            else:
                return redirect("password_reset_done")
    else:
        form = PasswordResetForm()
    return render(request=request, template_name="authentication/forgot_password.html", context={"form": form})

def reset_password(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None
        
    if user is not None and default_token_generator.check_token(user, token):
        if request.method == 'POST':
            new_password = request.POST.get('new_password')
            confirm_password = request.POST.get('confirm_password')
            if new_password == confirm_password:
                user.set_password(new_password)
                user.save()
                return redirect("password_reset_complete") 
            else:
                return render(request, 'authentication/password_reset.html', {'error_message': "Passwords do not match."})
        else:
            return render(request, 'authentication/password_reset.html', {'uidb64': uidb64, 'token': token})
    else:
        return render(request, 'authentication/password_reset.html', {'error_message': "The link is no longer valid."})

def feedback_view(request):
    if request.method == 'POST':
        form = FeedbackForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            feedback = form.cleaned_data['feedback']

            sender_email = 'predictlyfeedback@gmail.com'

            email_content = render_to_string('email_templates/user_feedback_email.html', {'email': email, 'feedback': feedback})

            send_mail(
                '✉️ New Customer Feedback',
                '',
                sender_email,
                ['predictlyfeedback@gmail.com'],
                html_message=email_content,
                fail_silently=False,
            )

            return JsonResponse({'message': 'Feedback submitted successfully'})
        else:
            return JsonResponse({'message': 'Invalid form data'}, status=400)
    else:
        form = FeedbackForm()

    return render(request, 'feedback_template.html', {'form': form})

@login_required  
def my_watchlist(request):
    if request.method == 'POST':
        form = StockForm(request.POST)
        if form.is_valid():
            stock = form.save(commit=False)
            stock.user = request.user 
            stock.save()
            messages.success(request, "Stock has been added!")
            return redirect('my_watchlist')
    else:
        form = StockForm()  
        ticker = Stock.objects.filter(user=request.user)  
        output = []

        for ticker_item in ticker:
            api_request = requests.get(
                f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker_item}&apikey={ALPHA_VANTAGE_API_KEY}")
            try:
                api = json.loads(api_request.content)
                if 'Global Quote' in api:
                    stock_data = api['Global Quote']
                    stock_data['pk'] = ticker_item.pk
                    stock_data['added_at'] = ticker_item.added_at
                    output.append(stock_data)
                else:
                    output.append({'symbol': str(ticker_item), 'error': 'Data not found: API limit reached.'})
            except Exception as e:
                output.append({'symbol': str(ticker_item), 'error': str(e)})
             
        fname = request.user.first_name.capitalize() if request.user.is_authenticated else ""
        
        return render(request, 'pages/my_watchlist.html', {'form': form, 'ticker': ticker, 'output': output, 'fname': fname,})

@login_required
def delete(request, stock_id):
    try:
        stock = Stock.objects.get(pk=stock_id)
        if stock.user == request.user: 
            stock.delete()
            messages.success(request, "Stock has been deleted!")
        else:
            messages.error(request, "You can only delete your own stocks!")
    except Stock.DoesNotExist:
        messages.error(request, "Stock not found!")
    
    return redirect('my_watchlist')

class LastTradePricesAPIView(View):
    def get(self, request, ticker_symbol=None):
        
        ticker_symbol = request.GET.get('ticker_symbol', 'AAPL')

        if not isinstance(ticker_symbol, str) or not ticker_symbol.strip():
            ticker_symbol = 'AAPL'

        start_date = '2010-01-01'
        end_date = datetime.datetime.today().strftime('%Y-%m-%d')
        
        # Fetch stock data using the services module
        stockdataframe, error_message = fetch_stock_data(ticker_symbol, start_date, end_date)
        
        # If we still don't have data after all retries, return error
        if stockdataframe is None or stockdataframe.empty:
            context = {
                'ticker_symbol': ticker_symbol, 
                'error_message': error_message or "Invalid Ticker or No Data Available",
            }
            return render(request, 'pages/last_trade_prices_template.html', context)
        
        try:
            closing_prices_plot = generate_closing_prices_plot(stockdataframe)
            raw_data_summary = stockdataframe.describe()
            
            # Get company info using the services module
            company_name, stock_exchange = get_company_info(ticker_symbol)
            
        except Exception as e:
            context = {
                'ticker_symbol': ticker_symbol,
                'error_message': f"Error processing data for {ticker_symbol}: {str(e)}"
            }
            return render(request, 'pages/last_trade_prices_template.html', context)
        
        if request.user.is_authenticated:
            fname = request.user.first_name
        else:
            fname = ""

        context = {
           'company_name': company_name,
            'stock_exchange': stock_exchange,
            'raw_data_summary': raw_data_summary,
            'closing_prices_plot': closing_prices_plot,
            'stockdataframe': stockdataframe.to_html(classes='table table-bordered table-striped'),
            'ticker_symbol': ticker_symbol, 
            'fname': fname,  
        }
        
        return render(request, 'pages/last_trade_prices_template.html', context)

class AutomatedCrossoverAPIView(APIView):
    def get(self, request, ticker_symbol=None):
        
        ticker_symbol = request.GET.get('ticker_symbol', 'AAPL')

        if not isinstance(ticker_symbol, str) or not ticker_symbol.strip():
            ticker_symbol = 'AAPL'

        start_date = '2010-01-01'
        end_date = datetime.datetime.today().strftime('%Y-%m-%d')

        # Fetch stock data using the services module
        stockdataframe, error_message = fetch_stock_data(ticker_symbol, start_date, end_date)
        
        # If we still don't have data after all retries, return error
        if stockdataframe is None or stockdataframe.empty:
            context = {
                'ticker_symbol': ticker_symbol, 
                'error_message': error_message or "Invalid Ticker or No Data Available",
            }
            return render(request, 'pages/crossover_data_template.html', context)

        try:
            # Get company info using the services module
            company_name, stock_exchange = get_company_info(ticker_symbol)

            # Calculate SMA dataframe using the services module
            sma_dataframe = calculate_sma_dataframe(stockdataframe)

            crossover_plot = generate_crossover_plot(sma_dataframe)
            cross_signal = determine_cross_signal(sma_dataframe)

        except ValueError as ve:
            context = {
                'ticker_symbol': ticker_symbol,
                'error_message': str(ve)
            }
            return render(request, 'pages/crossover_data_template.html', context)
        except Exception as e:
            context = {
                'ticker_symbol': ticker_symbol,
                'error_message': f"Error processing crossover data: {str(e)}"
            }
            return render(request, 'pages/crossover_data_template.html', context)

        fname = request.user.first_name if request.user.is_authenticated else ""

        context = {
            # 'company_name': company_name,
            'stock_exchange': stock_exchange,
            'crossover_plot': crossover_plot,
            'sma_dataframe': sma_dataframe.to_html(classes='table table-bordered table-striped'),
            'ticker_symbol': ticker_symbol, 
            'fname': fname, 
            'cross_signal': cross_signal, 
        }

        return render(request, 'pages/crossover_data_template.html', context)

class PredictionVsActualAPIView(APIView):
    def get(self, request, ticker_symbol=None):
                
        ticker_symbol = request.GET.get('ticker_symbol', 'AAPL')

        if not isinstance(ticker_symbol, str) or not ticker_symbol.strip():
            ticker_symbol = 'AAPL'

        start_date = '2010-01-01'
        end_date = datetime.datetime.today().strftime('%Y-%m-%d')

        # Fetch stock data using the services module
        stockdataframe, error_message = fetch_stock_data(ticker_symbol, start_date, end_date)

        # If we still don't have data after all retries, return error
        if stockdataframe is None or stockdataframe.empty:
            context = {
                'ticker_symbol': ticker_symbol,
                'error_message': error_message or "No data available for the specified ticker",
            }
            return render(request, 'pages/prediction_vs_actual_template.html', context)

        try:
            # Get company info using the services module
            company_name, stock_exchange = get_company_info(ticker_symbol)
            
            training_70, testing_30 = prepare_lstm_data(stockdataframe)
            
            # Try to get or train LSTM model using the services module
            model, scaler = get_or_train_lstm_model(training_70)

            try:
                # Make predictions using the services module
                y_test_flat, y_predicted_flat, input_data, scale_factor = make_predictions(
                    model, scaler, training_70, testing_30
                )
            except Exception as prediction_error:
                print(f"✗ Prediction failed with loaded model: {str(prediction_error)}")
                print("✓ Retrying with freshly trained model...")
                
                # Force retrain and try again
                model, scaler = get_or_train_lstm_model(training_70, force_retrain=True)
                y_test_flat, y_predicted_flat, input_data, scale_factor = make_predictions(
                    model, scaler, training_70, testing_30
                )
        
            dates = stockdataframe.index[int(len(stockdataframe) * 0.70):]
            
            if len(dates) != len(y_test_flat):
                dates = dates[:len(y_test_flat)]
            
            prediction_vs_actual_plot = generate_prediction_vs_actual_plot(dates, y_test_flat, y_predicted_flat)
                        
            # Generate Weekly Forecast Plot
            future_dates = pd.date_range(start=dates[-1], periods=8, freq='D')[1:]  # Next 7 days
            
            # Generate future predictions using the services module
            future_predictions = generate_future_predictions(model, input_data, scale_factor, days=7)

            weekly_forecast_plot = generate_weekly_forecast_plot(future_dates, future_predictions)
            
            future_trend = 'Bearish Trend' if future_predictions[0] > future_predictions[-1] else 'Bullish Trend'
            
        except ValueError as ve:
            context = {
                'ticker_symbol': ticker_symbol,
                'error_message': str(ve)
            }
            return render(request, 'pages/prediction_vs_actual_template.html', context)
        except Exception as e:
            context = {
                'ticker_symbol': ticker_symbol,
                'error_message': f"Error processing prediction data: {str(e)}"
            }
            return render(request, 'pages/prediction_vs_actual_template.html', context)
        
        context = {
            'stock_exchange': stock_exchange,
            'prediction_vs_actual_plot': prediction_vs_actual_plot.to_html(full_html=False),
            'weekly_forecast_plot': weekly_forecast_plot.to_html(full_html=False),
            'ticker_symbol': ticker_symbol,
            'future_trend': future_trend, 
        }

        return render(request, 'pages/prediction_vs_actual_template.html', context)
